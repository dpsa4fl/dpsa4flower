
"""Flower server."""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from dpsa4fl_bindings import (
    controller_api_new_state,
    controller_api_create_session,
    controller_api_start_round,
    controller_api_collect,
    controller_api_end_session,
    PyControllerState
)

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import configure, log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import Server, FitResultsAndFailures
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy, FedAvg


class DPSAServer(Server):
    """
    A flower server for federated learning with global differential privacy and
    secure aggregation. Uses the dpsa project infrastructure, see here
    for more information: https://github.com/dpsa-project/overview

    NOTE: This is intended for use with the DPSANumPyClient flower client.
    """ 

    def __init__(
        self,
        *,
        model_size: int,
        privacy_parameter: float,
        granularity: int,
        aggregator1_location: str,
        aggregator2_location: str,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None
    ) -> None:
        """
        Parameters
        ----------
        model_size: int
            The number of parameters of the model to be trained.
        privacy_parameter: float
            The desired privacy per learning step. One aggregation step will
            be `1/2*privacy_parameter^2` zero-concentrated differentially private
            for each client.
        granularity: int
            The resolution of the fixed-point encoding used for secure aggregation.
            A larger value will result in a less lossy representation and more
            communication and computation overhead. Currently, 16, 32 and 64 bit are
            supported.
        aggregator1_location: str
            Location of the first aggregator server in URL format including the port.
            For example, for a server running locally: "http://127.0.0.1:9991"
        aggregator2_location: str
            Location of the second aggregator server in URL format including the port.
            For example, for a server running locally: "http://127.0.0.1:9992"
        """

        # call dpsa4fl to create state object
        #
        # NOTE: this might fail, e.g. if servers are not running at the given locations
        try:
            self.dpsa4fl_state = controller_api_new_state(
                model_size,
                privacy_parameter,
                granularity,
                aggregator1_location,
                aggregator2_location,
            )
        except RuntimeError as err:
            print("=======================================")
            print("Could not initialize controller state.")
            print(f"Are aggregator servers running at {aggregator1_location} and {aggregator2_location}?")
            print("=======================================")
            print("")
            print(f"The original error message is: {err}")

        dpsa4fl_strategy = DPSAStrategyWrapper(
            strategy = strategy if strategy is not None else FedAvg(),
            dpsa4fl_state = self.dpsa4fl_state
        )

        super().__init__(client_manager=client_manager, strategy=dpsa4fl_strategy)

        # call dpsa4fl to create new session
        controller_api_create_session(self.dpsa4fl_state)
        
    """End the dpsa4fl session. Use at the end of training for graceful shutdown."""
    def __del__(self):
        # end session when we are done
        controller_api_end_session(self.dpsa4fl_state)

    """Perform a single round of federated averaging."""
    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        # Call dpsa4fl to start a new round
        controller_api_start_round(self.dpsa4fl_state)

        # The rest of the work is done by the `DPSAStrategyWrapper`
        # which is called in the server implementation of super.
        res = super().fit_round(server_round, timeout)

        return res


# flake8: noqa: E501
class DPSAStrategyWrapper(Strategy):
    """
    Configurable strategy implementation for federated learning with
    global differential privacy and secure aggregation. To be used together
    with the DPSAServer.
    """
    def __init__(self, *, strategy: Strategy, dpsa4fl_state: PyControllerState) -> None:
        """
        Parameters
        ----------
        strategy: Strategy
            The strategy to be wrapped. The wrapping replaces the `configure_fit` and
            `aggregate_fit` methods with ones that interact with the dpsa
            infrastructure. All other methods are copied from the input strategy.
        dpsa4fl_state: PyControllerState
            State object containing training parameters and server locations necessary
            for interoperation with dpsa. See `dpsa4fl_bindings.controller_api_new_state`
            for details.
        """ 
        super().__init__()
        self.strategy = strategy
        self.dpsa4fl_state = dpsa4fl_state
        self.parameters = Parameters(tensors=[], tensor_type="")

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        result = self.strategy.configure_fit(server_round, parameters, client_manager)
        def append_config(arg: Tuple[ClientProxy, FitIns]):
            client, fitins = arg

            # add the task_id into the config
            fitins.config['task_id'] = self.dpsa4fl_state.mstate.task_id

            return (client, fitins)

        self.parameters = parameters

        return list(map(append_config, result))


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # we do our custom aggregation here.

        print("Getting results from janus")
        collected: np.ndarray = controller_api_collect(self.dpsa4fl_state)
        print("Done getting results from janus, vector length is: ", collected.shape)

        # convert ndarray type
        flat_grad_array = collected.astype(np.float32)

        old_params_arrays = parameters_to_ndarrays(self.parameters)
        flat_old_params_array = old_params_arrays[0]

        # if old params is not flat, need to flatten
        if len(old_params_arrays) > 1:
            flat_old_params = [p.flatten('C') for p in old_params_arrays] #TODO: Check in which order we need to flatten here
            flat_old_params_array = np.concatenate(flat_old_params)

        # add gradient to current params
        flat_param_array = flat_old_params_array + flat_grad_array

        # encode again in params format
        parameters_aggregated = ndarrays_to_parameters([flat_param_array])

        # Aggregate custom metrics if aggregation fn was provided
        # the same as what happens in the FedAvg strategy
        metrics_aggregated = {}
        if hasattr(self.strategy, 'fit_metrics_aggregation_fn'):
            if self.strategy.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.strategy.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "Strategy has no fit_metrics_aggregation_fn")

        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.strategy.evaluate(server_round, parameters)

