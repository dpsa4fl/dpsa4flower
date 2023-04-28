# dpsa4flower
Server and client to use the [flower framework](https://flower.dev/) for differentially private federated learning with secure aggregation.

Made to be used with the [dpsa infrastructure](https://github.com/dpsa-project/overview), head there for an explanation of the system's participants and properties. Setup of additional aggregation servers is required, head [here](https://github.com/dpsa-project/dpsa4fl-testing-infrastructure) for instructions.

## Installation
To install, you require the following packages:
- python version 3.9 or higher.
- [poetry](https://python-poetry.org/) package manager for python

Once you have those, go ahead and clone this repository:
```
> git clone https://github.com/dpsa-project/dpsa4flower.git
```
Enter the new directory:
```
> cd dpsa4flower
```
Use poetry to create a virtualenv and install all dependencies:
```
> poetry shell
> poetry install
```
You're ready to use our classes now. Note that to actually run a learning task, you will need to provide locations at which two seperate dpsa4fl aggregation servers are running. See [here](https://github.com/dpsa-project/dpsa4fl-testing-infrastructure) for instructions or check out our example project.

## Example code
There is a [repo](https://github.com/dpsa-project/dpsa4fl-example-project) containing an example implementation learning the CIFAR task using a torch model, where learning is federated using flower with differential privacy and secure aggregation.

## Classes
This package exposes two classes, one for the server and one for the client.
### [`DPSAServer`](https://github.com/dpsa-project/dpsa4flower/blob/3f1becb09bb79dfe26f9ee959114cf6c36a31dbb/dpsa_flower/dpsa_server.py#L40)
The server class extends the flower [server class](https://flower.dev/docs/apiref-flwr.html#module-flwr.server) with the necessities for using DPSA for aggregation. It handles configuration of the aggregator servers, reshaping of the collected aggregation results, and redistributing the updates to the clients. Construction requires the following parameters:

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


### [`DPSANumPyClient`](https://github.com/dpsa-project/dpsa4flower/blob/3f1becb09bb79dfe26f9ee959114cf6c36a31dbb/dpsa_flower/dpsa_numpy_client.py#L19)
The client class implements the [`NumPyClient`](https://flower.dev/docs/apiref-flwr.html#numpyclient) interface provided by flower. It's a wrapper for existing `NumPyClient`s adding secure aggregation and differential privacy. The wrapped client is used for local training, results are then submitted to the secure aggregation infrastructure in an encrypted fashion. The constructor requires the following parameters:
 
- `max_privacy_per_round: float` The maximal zero-contentrated differential privacy budget allowed to be spent on a single round of training. If the selected server offers a weaker guarantee, no data will be submitted and an exception will be raised.
- `aggregator1_location: str` Location of the first aggregator server in URL format including the port. For example, for a server running locally: "http://127.0.0.1:9991"
- `aggregator2_location: str` Location of the second aggregator server in URL format including the port. For example, for a server running locally: "http://127.0.0.1:9992"
- `client: NumPyClient` The NumPyClient used for executing the local learning tasks.
- `allow_evaluate: bool` Evaluation is a privacy-relevant operation on the client dataset. If this flag is set to `False`, evaluation always reports infinite loss and zero accuracy to the server. Otherwise, the evaluation function of the wrapped client will be used and the results will be released to the server, potentially compromising privacy. Defaults to `False`.

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
