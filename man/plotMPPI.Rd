\name{plotMPPI}
\alias{plotMPPI}
\title{Plots the marginal posterior probability of inclusion (MPPI) for each predictor}
\usage{
  plotMPPI(x, threshold.model = 0.01,
    threshold.variable = 0.1, Figure = TRUE, cutoff = TRUE, useMC = FALSE)
}
\arguments{
  \item{x}{an object of class \code{ESS}.}

  \item{threshold.model}{either an integer representing the number of
      model to be retained in the list of best models, or a value
      defining the minimal model posterior probability for inclusion.}

  \item{threshold.variable}{threshold probability for selecting the
      most relevant predictors. This threshold can be calibrated by
      controlling the FDR
      using \code{\link{FDR.permutation}}.}

  \item{Figure}{if \code{TRUE} (by default) will generate the MPPI
      plot. If \code{FALSE} only information on the selected predictors
      will provided.}

  \item{cutoff}{if \code{TRUE} (by default) will plot an horizontal line
      representing the cut-off value indicating by the argument
      \code{threshold.variable}. If \code{FALSE} the cut-off value is not
      plotted.}

  \item{useMC}{if \code{TRUE}, use simple Monte Carlo estimation for the MPPI
      across all visited models.}
}

\value{The \code{plotMPPI} function returns information
  on the best models (i.e. those satisfying the
  \code{threshold.model} criterion) and on the most relevant
  predictors.
  (above \code{threshold.variable}).
\item{Rank}{the rank on the models selected.}
\item{nVisits}{number of times each model has been visited along the
    run.}
\item{ModSize}{number of predictors in each of the best models.}
\item{logCondPost}{the log conditional posterior for each model.}
\item{Jeffries}{Jeffrie's scale value for each model.}
\item{postProb}{posterior probability of each model.}
\item{modelName}{list of predictors in each of the best models.}
\item{modelPosInX}{position (in the predictor matrix) of the
    constituents of the best models.}
\item{var.TOP.MPI}{predictors with MPPI>\code{threshold.variable} and
  belonging to the best models.}
\item{var.MPI}{predictors which have a MPPI greater than
    \code{threshold.variable}.}
}


\description{ The \code{plotMPPI} function plots the marginal posterior
  probability of inclusion (MPPI) of each predictor.}


\examples{
modelY_Hopx <- example.as.ESS.object()
# To get a large plot 
# dev.new(width=13,height=6)
MPPI.Hopx <- plotMPPI(modelY_Hopx,threshold.model=20,threshold.variable=0.45)
print(MPPI.Hopx)
}
\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}
