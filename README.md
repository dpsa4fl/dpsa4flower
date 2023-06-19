# dpsa4flower
Server and client to use the [flower framework](https://flower.dev/) for differentially private federated learning with secure aggregation.

Made to be used with the [dpsa infrastructure](https://github.com/dpsa-project/overview), head there for an explanation of the system's participants and properties. Setup of additional aggregation servers is required, head to our [dpsa4fl infrastructore repo](https://github.com/dpsa-project/dpsa4fl-infrastructure) for instructions. There also is [an example implementation](https://github.com/dpsa-project/dpsa4fl-example-project).

## Installation
To install, you require the following packages:
- python version 3.9 or higher.
- [pip](https://pip.pypa.io/en/stable/) package installer for python

Once you have those, go ahead and install our package:
```
> pip install dpsa4flower
```
You're ready to use our classes now. Note that to actually run a learning task, you will need to provide locations at which two seperate dpsa4fl aggregation servers are running. See our [dpsa4fl infrastructore repo](https://github.com/dpsa-project/dpsa4fl-infrastructure) for instructions or check out our example project.

## Example code
There is a [repo containing an example implementation](https://github.com/dpsa-project/dpsa4fl-example-project) learning the CIFAR task using a torch model, where learning is federated using flower with differential privacy and secure aggregation.

## Classes
This package exposes two classes, one for the server and one for the client.
### `DPSAServer(model_size, privacy_parameter, granularity, aggregator1_location, aggregator2_location, client_manager, strategy)`

The [dpsa4flower server class](https://github.com/dpsa-project/dpsa4flower/blob/3f1becb09bb79dfe26f9ee959114cf6c36a31dbb/dpsa_flower/dpsa_server.py#L40) extends the [flower server class](https://flower.dev/docs/apiref-flwr.html#module-flwr.server) with the necessities for using DPSA for aggregation. It handles configuration of the aggregator servers, reshaping of the collected aggregation results, and redistributing the updates to the clients. Construction requires the following parameters as keyword arguments:

- `model_size: int` The number of parameters of the model to be trained.
- `privacy_parameter: float` The desired privacy per learning step. One aggregation step will
    be `1/2*privacy_parameter^2` zero-concentrated differentially private
    for each client.
- `granularity: int` The resolution of the fixed-point encoding used for secure aggregation.
    A larger value will result in a less lossy representation and more
    communication and computation overhead. Currently, 16, 32 and 64 bit are
    supported.
- `aggregator1_location: str` Location of the first aggregator server in URL format including the port.
    For example, for a server running locally: "http://127.0.0.1:9991"
- `aggregator2_location: str` Location of the second aggregator server in URL format including the port.
    For example, for a server running locally: "http://127.0.0.1:9992"
- `client_manager: flwr.server.ClientManager` A flower client manager to manage connected clients.
- `strategy: Optional[flwr.server.strategy.Strategy]` A flower strategy for the server to use. It will be wrapped replacing the `configure_fit` and `aggregate_fit` methods with ones that interact with the dpsa infrastructure.

An example construction of a dpsa4flower server object would look like this:
```python
dpsa4flower.DPSAServer(
        model_size = 62006,
        privacy_parameter = 30,
        granularity = 32,
        aggregator1_location = "http://127.0.0.1:9981",
        aggregator2_location = "http://127.0.0.1:9982",
        client_manager=flwr.server.SimpleClientManager(),
)
```
The created object can then be used to start a flower server using [`flwr.server.start_server`](https://flower.dev/docs/apiref-flwr.html#server-start-server) as usual.

### `DPSANumPyClient(max_privacy_per_round, aggregator1_location, aggregator2_location, client, allow_evaluate)`
The [dpsa4flower client class](https://github.com/dpsa-project/dpsa4flower/blob/3f1becb09bb79dfe26f9ee959114cf6c36a31dbb/dpsa_flower/dpsa_numpy_client.py#L19) implements the [`NumPyClient`](https://flower.dev/docs/apiref-flwr.html#numpyclient) interface provided by flower. It's a wrapper for existing `NumPyClient`s adding secure aggregation and differential privacy. The wrapped client is used for local training, results are then submitted to the secure aggregation infrastructure in an encrypted fashion. The constructor requires the following parameters as keyword arguments:
 
- `max_privacy_per_round: float` The maximal zero-contentrated differential privacy budget allowed to be spent on a single round of training. If the selected server offers a weaker guarantee, no data will be submitted and an exception will be raised.
- `aggregator1_location: str` Location of the first aggregator server in URL format including the port. For example, for a server running locally: "http://127.0.0.1:9991"
- `aggregator2_location: str` Location of the second aggregator server in URL format including the port. For example, for a server running locally: "http://127.0.0.1:9992"
- `client: flower.client.numpy_client.NumPyClient` The NumPyClient used for executing the local learning tasks.
- `allow_evaluate: bool` Evaluation is a privacy-relevant operation on the client dataset. If this flag is set to `False`, evaluation always reports infinite loss and zero accuracy to the server. Otherwise, the evaluation function of the wrapped client will be used and the results will be released to the server, potentially compromising privacy. Defaults to `False`.

An example construction of a dpsa4flower client object would look like this:
```python
dpsa4flower.DPSANumPyClient(
    max_privacy_per_round = 30,
    aggregator1_location = "http://127.0.0.1:9981",
    aggregator2_location = "http://127.0.0.1:9982",
    client = FlowerClient()
)
```
where `FlowerClient` is some [`NumPyClient`](https://flower.dev/docs/apiref-flwr.html#numpyclient) of your choice. It can then be started using [`flwr.client.start_numpy_client`](https://flower.dev/docs/apiref-flwr.html#flwr.client.start_numpy_client) as usual.

## What's going on
When using our classes in the setup described (and used in the example project), the training procedure takes place as described in this diagram:


```
                                             gradient sum
          ┌───────────────────────────────────────────────────────────────────────────────┐
          │                             (differentially private)                          │
          │                                                                               │
          │                                                                               │
          │                                                                               │
          │                                                                               │
          │             gradient shares                                                   │
          │              (ciphertext)                                                     │
  ┌───────▼─────────┬────────────────────┐                                                │
  │ DPSANumPyClient │                    │ ┌──────────────┐                               │
  └─────────────────┴────────────────┐   └─►              │                               │
                                     │     │ Aggregator 1 ├───┐                           │
                                 ┌───)─────►              │   │                           │
                                 │   │     └──────────────┘   │                   ┌───────┴───────┐
          .                      │   │                        │    gradient sum   │               │
          .                      │   │                        ├───────────────────►  DPSAServer   │
          .                      │   │                        │  (differentially  │               │
                                 │   │     ┌──────────────┐   │      private)     └───────────────┘
                                 │   └─────►              │   │
                                 │         │ Aggregator 2 ├───┘
  ┌─────────────────┬────────────┘   ┌─────►              │
  │ DPSANumPyClient │                │     └──────────────┘
  └─────────────────┴────────────────┘
                     gradient shares
                      (ciphertext)



     flower clients                     dpsa4fl infrastructure                     flower server
     --------------                     ----------------------                     -------------
compute gradients locally          checks if clipping was done properly,        collects aggregate,
   on sensitive data,               computes aggregate on ciphertext,           distributes updates
clip to norm 1 and submit          adds noise for differential privacy.         back to the clients
                                     ciphertext can not be decrypted
                                     if the servers don't collaborate
```
