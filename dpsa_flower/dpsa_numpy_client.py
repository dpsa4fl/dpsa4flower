
#
# Written as part of DPSA-project, by @MxmUrw, @ooovi
#
# Based on `dpfedavg_numpy_client.py`.
#

"""Wrapper for configuring a Flower client for usage with DPSA."""

from typing import Dict, Tuple

import numpy as np

from flwr.client.numpy_client import NumPyClient
from flwr.common.typing import Config, NDArrays, NDArray, Scalar

from dpsa4fl_bindings import client_api_new_state, client_api_submit, client_api_get_privacy_parameter

class DPSANumPyClient(NumPyClient):
    """
    A flower client for federated learning with global differential privacy and
    secure aggregation. Uses the dpsa project infrastructure, see here
    for more information: https://github.com/dpsa-project/overview

    NOTE: This is intended for use with the DPSAServer flower server.

    Attributes
    ----------
    privacy_spent: float
        Zero-concentrated Differential Privacy of this client's data
        spent since DPSANumPyClient object construction. Accumulates over
        training rounds.
    """

    def __init__(
        self,
        max_privacy_per_round: float,
        aggregator1_location: str,
        aggregator2_location: str,
        client: NumPyClient,
        allow_evaluate = false
    ) -> None:
        """
        Parameters
        ----------
        max_privacy_per_round: float
            The maximal zero-contentrated differential privacy budget allowed to
            be spent on a single round of training. If the selected server offers
            a weaker guarantee, no data will be submitted and an exception will be raised.
        aggregator1_location: str
            Location of the first aggregator server in URL format including the port.
            For example, for a server running locally: "http://127.0.0.1:9991"
        aggregator2_location: str
            Location of the second aggregator server in URL format including the port.
            For example, for a server running locally: "http://127.0.0.1:9992"
        client: NumPyClient
            The NumPyClient used for executing the local learning tasks.
        allow_evaluate: bool
            Evaluation is a privacy-relevant operation on the client dataset. If this flag
            is set to `false`, evaluation always reports infinite loss and zero accuracy to
            the server. Otherwise, the evaluation function of the wrapped client will be used
            and the results will be released to the server, potentially compromising privacy.
            Defaults to false.
        """
        super().__init__()
        self.max_privacy_per_round = max_privacy_per_round
        self.client = client
        self.dpsa4fl_client_state = client_api_new_state(
            aggregator1_location,
            aggregator2_location,
        )
        self.shapes = None
        self.split_indices = None
        self.privacy_spent = 0

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        return self.client.get_properties(config)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return self.client.get_parameters(config)

    def reshape_parameters(self, parameters: NDArrays) -> NDArrays:
        """
        Reshape parameters given as a (1,)-NDArray into the Client's model shape,
        as given in the `shapes` attribute. Used for recovering the shape of a
        gradient that originated from the DPSA aggregation server and was hence in
        single-vector shape.

        Parameters
        ----------
        parameters: NDArrays
            The parameters to be reshaped. Expected to be (1,)-shaped.

        Returns
        -------
        parameters: NDArrays
            The input parameters reshaped to match this Client's `shapes` attribute.
            
        """
        # update parameter shapes
        # if we are in first round (self.shapes is None), then we don't need to reshape.
        # But if we are in any following round, then we need to take our previous shapes
        # and lengths and reshape the `parameters` argument accordingly
        if (self.shapes is not None) and (self.split_indices is not None):
            assert len(self.split_indices) + 1 == len(self.shapes), "Expected #indices = #shapes - 1"

            print("In follow-up round, reshaping. length of params is: ", len(parameters))
            assert len(parameters) == 1, "Expected parameters to have length 1!"

            single_array = parameters[0]
            print("Found single ndarray of shape ", single_array.shape, " and size ", single_array.size)
            # assert single_array.shape == (,), "Wrong ndarray shape!"

            # debug: print highest element
            print("(reshape) highest nan element is: ", np.amax(single_array))
            print("(reshape) highest nonnan element is: ", np.nanmax(single_array))

            # split and reshape
            arrays = np.split(single_array, self.split_indices)
            print("After splitting, have ", len(arrays), " arrays")

            arrays = [np.reshape(a,s) for (a,s) in zip(arrays, self.shapes)]
            print("Now have the following shapes:")
            for a in arrays:
                print(a.shape)

            # check that we have our value at position 449
            rval1 = np.around(arrays[0], decimals = 2)
            print("in array at 449 have: ", rval1[0:1, 0:1, 0:1, 0:1])
            print(rval1)

            # change parameters to properly shaped list of arrays
            parameters = arrays

        else:
            print("In first round, not reshaping.")

        return parameters

    def flatten_parameters(self, params: NDArrays) -> NDArray:
        """
        Reshape the input parameters into a (1,)-NDArray to prepare
        them for sumbission to the DPSA infrastructure.

        Parameters
        ----------
        params: NDArrays
            The parameters to be flattened. Flattening in row-major order.

        Returns
        -------
        flat_param_vector: NDArray
            The input parameters reshaped to one long (1,)-NDArray.
        """
        
        # print param shapes
        print("The shapes are:")
        for p in params:
            print(p.shape)

        # flatten params before submitting
        self.shapes = [p.shape for p in params]
        flat_params = [p.flatten('C') for p in params] #TODO: Check in which order we need to flatten here
        p_lengths = [p.size for p in flat_params]

        # loop
        # (convert p_lengths into indices because ndarray.split takes indices instead of lengths)
        split_indices = []
        current_index = 0
        for l in p_lengths:
            split_indices.append(current_index)
            current_index += l
        split_indices.pop(0) # need to throw away first element of list
        self.split_indices = split_indices


        flat_param_vector = np.concatenate(flat_params)


        print("vector length is: ", flat_param_vector.shape)

        # debug: print highest element
        print("highest nan element is: ", np.amax(flat_param_vector))
        print("highest nonnan element is: ", np.nanmax(flat_param_vector))

        return flat_param_vector

    def fit(
        self, params0: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the provided parameters using the locally held dataset. After
        training locally using the wrapped NumPyClient, the gradients are
        clipped to norm 1 and submitted to the DPSA infrastructure for secure
        aggregation of all clients' gradients and noising for differential privacy.

        Parameters
        ----------
        params0 : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.

        Returns
        -------
        parameters : NDArrays
            The locally updated model parameters.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """ 

        # get current task_id
        task_id = config['task_id']

        #reshape params
        params0 = self.reshape_parameters(params0)

        # train on data
        params, i, d = self.client.fit(params0, config)

        # compute gradient
        grad = [np.subtract(p, p0) for (p , p0) in zip(params,params0)]

        flat_grad_vector = self.flatten_parameters(grad)

        # truncate if norm >= 1
        norm = np.linalg.norm(flat_grad_vector)
        print("norm of vector is: ", norm)
        if norm >= 1:
            print("Need to scale vector")
            flat_grad_vector = flat_grad_vector * (1/(norm + 0.01))
            norm = np.linalg.norm(flat_grad_vector)
            print("now norm of vector is: ", norm)

        # get server privacy parameter
        eps = client_api_get_privacy_parameter(self.dpsa4fl_client_state, task_id)
        
        if eps > self.max_privacy_per_round:
            raise Exception("DPSAClient requested at least " + str(self.max_privacy_per_round) + " privacy but server supplied only " + str(eps))
        else:
            # log privacy loss
            self.privacy_spent += eps

            # submit data to janus
            client_api_submit(self.dpsa4fl_client_state, task_id, flat_grad_vector)

            # return empty, parameter update needs to be retrieved from janus
            return [], i, d

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        if self.allow_evaluate:
            parameters = self.reshape_parameters(parameters)
            return self.client.evaluate(parameters, config)
        else:
            return float('inf'), 1, {"accuracy": 0}
