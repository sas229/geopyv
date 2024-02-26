"""

Chain module for geopyv.

"""
import logging
import numpy as np
import geopyv as gp
from geopyv.object import Object
from alive_progress import alive_bar

log = logging.getLogger(__name__)


class ChainBase(Object):
    """
    Chain base class to be used as a mixin.
    """

    def __init__(self):
        super().__init__(object_type="Chain")
        """

        Chain base class initialiser.

        """

    def kde(
        self,
        *,
        axis=True,
        xlim=None,
        ylim=None,
        show=True,
        block=True,
        save=None,
    ):
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        fig, ax = gp.plots.kde_chain(
            data=self.data,
            axis=axis,
            xlim=xlim,
            ylim=ylim,
            show=show,
            block=block,
            save=save,
        )

    def _report(self, msg, error_type):
        if msg and error_type != "Warning":
            log.error(msg)
        elif msg and error_type == "Warning":
            log.warning(msg)
            return True
        if error_type == "ValueError" and msg:
            raise ValueError(msg)
        elif error_type == "TypeError" and msg:
            raise TypeError(msg)
        elif error_type == "IndexError" and msg:
            raise IndexError(msg)


class Chain(ChainBase):
    def __init__(
        self,
        *,
        field=None,
        sample_no=100000,
        ID="",
    ):
        """
        Initialisation of geopyv Bayes object.

        Parameters
        ----------
        field : gp.field.Field object
            The base field for Bayesian inference.
        sample_no : int
            Number of accepted samples in final chain.
        chain_no : int
            Number of chains.

        Attributes
        ----------
        data : dict
            Data object containing all settings and results.
            See the data structure :ref:`here <bayes_data_structure>`.
        solved : bool
            Boolean to indicate if the bayes has been solved.

        """
        self._initialised = False

        # Check inputs.
        self._report(
            gp.check._check_type(field, "field", [gp.field.Field]), "TypeError"
        )
        check = gp.check._check_type(sample_no, "sample_no", [int])
        if check:
            try:
                sample_no = int(sample_no)
                self._report(
                    gp.check._conversion(sample_no, "sample_no", int),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_range(sample_no, "sample_no", 1), "ValueError")
        check = gp.check._check_type(ID, "ID", [str])
        if check:
            try:
                ID = str(ID)
            except Exception:
                self._report(check, "TypeError")

        # Attribute.
        self._ID = ID
        self.solved = False
        self._unsolvable = False
        self._field = field
        self._sample_no = sample_no
        self._k_c = np.zeros(self._sample_no)
        self._s_c = np.zeros(self._sample_no)
        self._a_c = np.zeros(self._sample_no)

        # Store.
        self.data = {
            "type": "Chain",
            "ID": self._ID,
            "solved": self.solved,
            "unsolvable": self._unsolvable,
            "sample_no": self._sample_no,
            "k_c": self._k_c,
            "s_c": self._s_c,
        }

        self._initialised = True

    def solve(
        self,
        *,
        model=None,
        state=None,
        parameters=None,
        variances=None,
        noise=None,
        prior=None,
        ext_works=None,
        verbose=True,
    ):
        """
        Method to solve for the bayes.

        Returns
        -------
        solved : bool
            Boolean to indicate if the particle instances have been solved.
        """
        self._model = model
        self._state = state
        self._parameters = parameters
        self._variances = variances
        self._noise = noise
        self._prior = prior
        self._ext_works = ext_works
        self.data.update(
            {
                "model": self._model,
                "state": self._state,
                "parameters": self._parameters,
                "variances": self._variances,
                "noise": self._noise,
                "prior": self._prior,
                "ext_works": self._ext_works,
            }
        )
        self._field.stress(model=model, state=state, parameters=parameters)
        ll = self._evaluation()
        accepted = 0
        rejected = 0
        s = parameters[5]
        k = parameters[6]
        with alive_bar(
            self._sample_no,
            dual_line=True,
            bar="blocks",
            title="MCMC MH sampling...",
            disable=not verbose,
        ) as bar:
            while accepted < self._sample_no:
                s_star = np.random.normal(loc=s, scale=self._variances[0])
                k_star = np.random.normal(loc=k, scale=self._variances[1])
                if not self._prior_check(s_star=s_star, k_star=k_star):
                    rejected += 1
                    bar.text(
                        ("Iteration: {},\tAccepted Ratio: {}\t").format(
                            accepted + rejected,
                            round(accepted / (accepted + rejected), 2),
                        )
                    )
                    continue
                parameters[5] = s_star
                state[1] = s_star
                parameters[6] = k_star
                self._field.stress(model=model, state=state, parameters=parameters)
                ll_star = self._evaluation()
                Hr = ll / ll_star
                if 1 < Hr or np.random.uniform(low=0.0, high=1.0) <= Hr:
                    s = s_star
                    k = k_star
                    ll = ll_star
                    self._s_c[accepted] = s
                    self._k_c[accepted] = k
                    self._a_c[accepted] = accepted / max(accepted + rejected, 1)
                    accepted += 1
                    bar()
                    bar.text(
                        ("Iteration: {},\tAccepted Ratio: {}\t").format(
                            accepted + rejected,
                            round(accepted / (accepted + rejected), 2),
                        )
                    )
                else:
                    rejected += 1
                    bar.text(
                        ("Iteration: {},\tAccepted Ratio: {}\t").format(
                            accepted + rejected,
                            round(accepted / (accepted + rejected), 2),
                        )
                    )
        log.info(
            ("Iteration: {},\tAccepted Ratio: {}\t").format(
                accepted + rejected, round(accepted / (accepted + rejected), 2)
            )
        )
        self.solved = True
        self.data.update(
            {
                "solved": self.solved,
                "k_c": self._k_c,
                "s_c": self._s_c,
                "a_c": self._a_c,
            }
        )

    def _evaluation(self):
        int_works = self._field._works
        N = np.shape(int_works)[0]
        ll = (
            -0.5 * N * np.log(2 * np.pi * self._noise)
            - 0.5 * N * self._noise**2
            - 2 * np.sum((int_works - self._ext_works) ** 2)
        )
        return ll

    def _prior_check(self, s_star, k_star):
        return (
            (k_star > self._prior[1, 0])
            * (k_star < self._prior[1, 1])
            * (s_star > self._prior[0, 0])
            * (s_star < self._prior[0, 1])
        )


class ChainResults(ChainBase):
    """

    ChainResults class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Field object.


    Attributes
    ----------
    data : dict
        geopyv data dict from Chain object.

    """

    def __init__(self, data):
        """Initialisation of geopyv ChainResults class."""
        self.data = data
        self._k_c = self.data["k_c"]
        self._s_c = self.data["s_c"]
        self._a_c = self.data["a_c"]
        self._sample_no = int(self.data["sample_no"])
