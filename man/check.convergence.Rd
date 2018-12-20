\name{check.convergence}

\alias{check.convergence}

\title{Diagnostic plots for the evaluation of the convergence of the
  algorithm}

\usage{
  check.convergence(x, which = c(1L:2L), nsplit = 10, nbloc = 4,
               ask = prod(par("mfcol")) < length(which) && dev.interactive())
}

\arguments{
  \item{x}{an object of class \code{ESS}}.
  
  \item{which}{if a subset of the plots is required, specify a subset of the
          numbers '1:2'.}
  
  \item{nsplit}{number of splits: the number of sweep intervals to consider.}

  \item{nbloc}{number of moving windows to plot.}

  \item{ask}{ask: logical; if 'TRUE', the user is _ask_ed before each plot, see
          'par(ask=.)'}	}
 
\value{The function \code{check.convergence} produces plots to
    assess the stability of the density estimates of the log posterior
    distribution. Two sets of graphs are plotted: the first set
    displays the density estimates of the log posterior distribution
    based on (i) all the sweep (ii) first half of the sweeps (iii)
    second half of the sweeps. The second plot represents similar
    density estimates at different stages of the MCMC procedure
    according to a sliding window whose characteristics are defined by
    the number of splits \code{nsplit} (i.e. the number of sweep
    intervals to consider) and number of blocks \code{nbloc}
    (i.e. the number of density estimates to plot).}


  \description{ The \code{\link{check.convergence}} function provides
    two plots (selectable by \code{which}) to investigate the convergence of the posterior
    distribution towards the target distribution at different stages
    of the algorithm.}


  \examples{
# load an ESS object
modelY_Hopx <- example.as.ESS.object()
check.convergence(modelY_Hopx,nsplit=10,nbloc=4)
check.convergence(modelY_Hopx,nsplit=5,nbloc=4)
check.convergence(modelY_Hopx,nsplit=5,nbloc=2)
}
 \author{ Benoit
    Liquet, \email{b.liquet@uq.edu.au},\cr Marc
    Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr Leonardo Bottolo
    \email{l.bottolo@imperial.ac.uk},\cr Gianluca Campanella
    \email{g.campanella11@imperial.ac.uk} }
