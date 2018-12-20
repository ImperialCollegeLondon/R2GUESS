\name{Analysis.permutation}
\alias{Analysis.permutation}
\title{Computing the FDR-controlled level for the significance of the MPPI}
\usage{
  Analysis.permutation(x, Npermut, namePermut, threshold = 0.05,
  path.output,  number.cutoff=200)
}
\arguments{ 
  \item{x}{an object of class "ESS".}

  \item{Npermut}{number of permutation runs to base the FDR
      calculations on. If the number exceeds the number of actual
      permutations then the calculation will stop and the function
      will return an error.}

  \item{namePermut}{name specifying the file name stem locating where
      results from the preliminary permutation runs were stored.}

  \item{threshold}{numeric value specifying the desired the FDR level.}

\item{path.output}{path directory containing the output files of
    the permuted runs.}  
  \item{number.cutoff}{numeric value specifying the number of points
      on which to be base the FDR estimation.}
}
\value{ \code{\link{Analysis.permutation}} returns a list
      containing the following fields:
\item{cutoff.MPI}{The MPPI threshold to control empirical FDR at a specified level.}
\item{cutoff_int}{The linearly interpolation (across the
    \code{number.cutoff} points) cut-off value of MPPI exactly controlling
    the FDR at the specified level.}
\item{cutoff_St}{the vector of cutoff values investigated (containing
    \code{number.cutoff} elements).}
\item{FDR_emp}{Empirical FDR corresponding to the \code{cutoff.MPI} value.}
\item{FDR_emp_int}{Empirical FDR value estimated by linear interpolation for the \code{cutoff_int} value.}
\item{FDR_emp_St}{A vector of empirical FDR values computed by
    linear interpolation for each \code{cutoff_St} values
    investigated.}
}



\description{Reads outputs from a preliminary set of permutation runs
  of GUESS mimicking the null hypothesis of no association (shuffling
  the responses). Results will subsequently be used to derive the
  cut-off values for the Marginal Posterior Probability Inclusion of 
  (MPPI ) ensuring an empirical FDR control at a user-defined
  level. Several cut-off values will be investigated and, for each, the
  corresponding empirical FDR will be returned. The cut-off value
  providing the closest FDR estimate to the desired level will be
  retained. Exact match to the desired level will be achieved by
  linear interpolation. For flexibility, \code{Analysis.permutation}
  only performs the calculation of the FDR-controlled cut-off
  value based on a preliminary set of permuted runs of GUESS, while
  the function \code{\link{FDR.permutation}} - which is based on the
  same procedure - also includes the permutations.  }


\examples{
\dontrun{
path.output.perm <- tempdir()
path.input.perm <-path.output.perm

cutoff.1 <- FDR.permutation(x=modelY_Hopx,Npermut=6,start.counter=1,
          path.output=path.output.perm,path.input=path.input.perm,nbcpu=3)

cutoff.2 <- FDR.permutation(x=modelY_Hopx,Npermut=6,start.counter=7,
          path.output=path.output.perm,path.input=path.input.perm,nbcpu=3)

namePermut <- "Permut-Example-GUESS-Y-Hopx"

cutoff.pool <-Analysis.permutation(modelY_Hopx,namePermut=namePermut,
      Npermut=9,threshold=0.05,path.output=path.output.perm,number.cutoff=10)
}
}
\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}

\seealso{
  \code{\link{FDR.permutation}}
}
