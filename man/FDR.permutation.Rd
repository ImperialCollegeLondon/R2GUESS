\name{FDR.permutation}
\alias{FDR.permutation}
\title{Performs parallel permuted runs of \code{R2GUESS} and returns
  the empirical FDR-controlled level for the significance of the MPPI}
\usage{
  FDR.permutation(x,path.input = NULL, Npermut, start.counter=1,
     path.output = NULL, threshold = 0.05, nbcpu = NULL, number.cutoff=200)
}
\arguments{
  \item{x}{an object of class \code{ESS}.}
  \item{path.input}{path to the directory containing the permuted re-samples of
      the \code{Y} matrix. By default (=\code{NULL}), the permuted
      outcomes are stored in the same directory as the results from
      the original \code{R2GUESS} run
      (\code{path.input}).}

  \item{path.output}{path to the directory in which results from
      permuted data are stored. By default (=NULL), these are saved in
      the directory where results from the original \code{GUESS} run
      were stored (\code{path.output}).}
  \item{Npermut}{number of permutations to run.}
  \item{start.counter}{defines the integer from which to start labelling permutation runs.}
  \item{threshold}{numeric value specifying the desired FDR level.}

  \item{nbcpu}{integer indicating the number of CPUs to use for the
      permutation procedure. This number has to be lower than the
      number of cores available on the platform. By default (=NULL),
      the function uses a single core.}
  \item{number.cutoff}{numeric value specifying the number of points
      on which to base the FDR estimation.}

}

\value{\code{FDR.permutation} generates permutation re-samples from
  the original \code{Y} matrix and generates for each permutation
  standard \code{R2GUESS} output files. Sets of results can be separately
  analysed using \code{\link{Analysis.permutation}}. The function also
  returns a list containing the following fields:
\item{cutoff.MPI}{the MPPI threshold to control empirical FDR at a specified level.}
\item{cutoff_int}{the linearly interpolated (across the
    \code{number.cutoff} points) cut-off value of MPPI exactly controlling
    the FDR at the specified level.}
\item{cutoff_St}{the vector of cut-off values investigated (containing
    \code{number.cutoff} elements).}
\item{FDR_emp}{empirical FDR corresponding to the \code{cutoff.MPI} value.}
\item{FDR_emp_int}{empirical FDR value estimated by linear interpolation for the \code{cutoff_int} value.}
\item{FDR_emp_St}{a vector of empirical FDR values computed by
    linear interpolation for each \code{cutoff_St} values investigated.}}

\description{ \code{FDR.permutation} will first run GUESS in
  parallel (over CPUs) for several permutations of the \code{Y} matrix
  mimicking the null hypothesis of no association. Results from the
  permutation procedure will subsequently be used to derive the
  cut-off values for the Marginal Posterior Probability of Inclusion
  (\code{MPPI}) ensuring an empirical FDR control at a user-defined
  level. Several cut-off values will be investigated and, for each, the
  corresponding empirical FDR will be returned. The cut-off value
  providing the closest FDR estimate to the desired level will be
  retained. Exact match to the desired level will be achieved by linear
  interpolation. The latter is based on the same calculation as in
  \code{\link{Analysis.permutation}}.}


\examples{
\dontrun{
modelY_Hopx <- example.as.ESS.object()
path.output.perm <- tempdir()
path.input.perm <-path.output.perm

cutoff <- FDR.permutation(x=modelY_Hopx,Npermut=100,start.counter=1,
           path.output=path.output.perm,path.input=path.input.perm,nbcpu=3)
 } }



\author{ Benoit Liquet \email{b.liquet@uq.edu.au},\cr
  Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr Leonardo
  Bottolo \email{l.bottolo@imperial.ac.uk},\cr Gianluca Campanella
  \email{g.campanella11@imperial.ac.uk} }

\seealso{
  \code{\link{Analysis.permutation}}
}
