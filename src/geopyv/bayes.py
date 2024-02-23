"""

Bayes module for geopyv.

"""
import logging
import numpy as np
import geopyv as gp
from geopyv.object import Object
from alive_progress import alive_bar

log = logging.getLogger(__name__)


class BayesBase(Object):
    """
    Bayes base class to be used as a mixin.
    """

    def __init__(self):
        super().__init__(object_type="Bayes")
        """

        Bayes base class initialiser.

        """

    def kde(
        self,
        *,
        chain_id=0,
        true=None,
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
            data=self.data["chains"][chain_id].data,
            R_t=self.data["results"]["R_t"],
            true=true,
            axis=axis,
            xlim=xlim,
            ylim=ylim,
            show=show,
            block=block,
            save=save,
        )

    def convergence(
        self,
        *,
        true=None,
        axis=True,
        klim=None,
        slim=None,
        show=True,
        block=True,
        save=None,
    ):
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        fig, ax = gp.plots.convergence_bayes(
            data=self.data,
            true=true,
            axis=axis,
            klim=klim,
            slim=slim,
            show=show,
            block=block,
            save=save,
        )

    def autocorrelation(
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

        fig, ax = gp.plots.convergence_bayes(
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


class Bayes(BayesBase):
    def __init__(
        self,
        *,
        chains=[],
        ID="",
    ):
        """
        Initialisation of geopyv Bayes object.

        Parameters
        ----------
        chains : list of gp.bayes.Chain objects.
            The MCMC MH sampling chains.
        ID :

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
        check = gp.check._check_type(ID, "ID", [str])
        if check:
            try:
                ID = str(ID)
            except Exception:
                self._report(check, "TypeError")

        # Attribute.
        self._ID = ID
        self.solved = True  # False
        self._unsolvable = False
        self._chains = chains
        self._chain_no = np.shape(self._chains)[0]
        sn = np.zeros(self._chain_no)
        for i in range(self._chain_no):
            sn[i] = self._chains[i]._sample_no
        self._sample_no = int(np.min(sn))

        # Store.
        self.data = {
            "type": "Bayes",
            "ID": self._ID,
            "solved": self.solved,
            "chains": self._chains,
            "chain_no": self._chain_no,
            "sample_no": self._sample_no,
        }

        self._initialised = True

    def solve(self, *, autocorlim=200):
        """
        Method to solve for the bayes.

        Returns
        -------
        solved : bool
            Boolean to indicate if the particle instances have been solved.
        """
        self._autocorlim = autocorlim
        samples = np.zeros((self._chain_no, self._sample_no, 2))
        for i in range(self._chain_no):
            samples[i, :, 0] = self._chains[i]._k_c
            samples[i, :, 1] = self._chains[i]._s_c
        self._convergence(samples)
        self._autocorrelation(samples)
        # self._process()
        self.data.update(
            {
                "results": {
                    "chain_means": self._chain_means,
                    "chain_vars": self._chain_vars,
                    "R": self._R,
                    "R_t": self._R_t,
                    "autocorrelation": self._autocorrelation,
                    "A_t": self._A_t,
                }
            }
        )

    def _convergence(self, samples):
        self._chain_means = (
            np.cumsum(samples, axis=1)
            / np.arange(1, self._sample_no + 1)[:, np.newaxis]
        )
        self._chain_vars = np.zeros(np.shape(samples))
        self._chain_vars[:, 1:] = (
            np.cumsum((samples - self._chain_means) ** 2, axis=1)[:, 1:]
            / np.arange(2, self._sample_no + 1)[:, np.newaxis]
        )
        mean = np.mean(self._chain_means, axis=0)
        W = np.mean(self._chain_vars, axis=0)
        B = (
            np.arange(1, self._sample_no + 1)[:, np.newaxis]
            * np.sum((self._chain_means - mean) ** 2, axis=0)
            / (self._chain_no - 1)
        )
        V = (
            np.arange(self._sample_no)[:, np.newaxis]
            * W
            / np.arange(1, self._sample_no + 1)[:, np.newaxis]
            + B / np.arange(1, self._sample_no + 1)[:, np.newaxis]
        )
        self._R = np.sqrt(V[1:] / W[1:])  # R incalculable for index zero.
        try:
            self._R_t = (
                np.argwhere(np.prod((abs(self._R - 1) < 0.01), axis=1))[0][0] + 1
            )
        except Exception:
            self._R_t = None

    def _autocorrelation(self, samples):
        self._autocorrelation = np.zeros((self._autocorlim, self._chain_no, 2))
        with alive_bar(
            self._autocorlim,
            dual_line=True,
            bar="blocks",
            title="Autocorrelation...",
        ) as bar:
            for k in range(self._autocorlim):
                asum = 0
                for s in range(self._sample_no - k):
                    asum += (samples[:, s] - self._chain_means[:, -1]) * (
                        samples[:, s + k] - self._chain_means[:, -1]
                    )
                self._autocorrelation[k] = asum / (
                    (self._sample_no - k) * self._chain_vars[:, -1]
                )
                bar()
        # try:
        self._A_t = (
            np.argwhere(np.prod((abs(self._autocorrelation) < 0.1), axis=1))[0][0] + 1
        )
        # except Exception:
        # self._A_t = None
        print(self._A_t)


class BayesResults(BayesBase):
    """

    BayesResults class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Field object.


    Attributes
    ----------
    data : dict
        geopyv data dict from Field object.

    """

    def __init__(self, data):
        """Initialisation of geopyv BayesResults class."""
        self.data = data
        self._chains = self.data["chains"]
