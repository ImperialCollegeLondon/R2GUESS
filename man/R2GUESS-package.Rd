\docType{package} \name{R2GUESS-package} \alias{R2GUESS-package}
\title{Sparse Bayesian variable selection method for linear regression
  based analysis of possibly multivariate outcomes.}
\description{ R2GUESS is an R package that runs the GUESS code, a
  computationally optimised C++ implementation of a fully Bayesian
  variable selection approach (Evolutionary Stochastic Search; ESS
  algorithm) that can analyse single and multiple responses in an
  integrated way and can scale to genome-wide datasets. The multi-SNP
  model underlying GUESS seeks for the best combinations of SNPs to
  predict the (possibly multivariate) outcome(s). The program uses
  packages from the GNU Scientific Library (GSL) and offers the
  possibility to re-route computationally intensive linear algebra
  operations towards the Graphical Processing Unit (GPU) through the
  use of proprietary CULA-dense library.  The use of GPU-based
  numerical libraries implies extensive data transfer between the
  memory/CPU and the GPU, which, in turn, can be computationally
  expensive. Consequently, for smaller data sets (as the example
  provided in the package) in which matrix operations are not
  rate-limiting, the CPU version of GUESS may be more computationally
  efficient. To ensure an optimal use of the algorithm, and to
  enable running GUESS on non-CULA compatible systems, the call to
  GPU-based calculations within GUESS can be switched off through a
  single argument, and will automatically be disabled in non
  CULA-compatible systems.  Extensive documentation of the source C++
  code is available at
  \url{http://www.bgx.org.uk/software/GUESS_Doc_short.pdf}.

  The current manual details most features of the GUESS algorithm and
  focuses on the built-in scripts enabling easy runs and automatic
  post-processing of outputs from GUESS.}

\author{ Benoit Liquet,
  \email{b.liquet@uq.edu.au},\cr Marc Chadeau-Hyam
  \email{m.chadeau@imperial.ac.uk},\cr Leonardo Bottolo
  \email{l.bottolo@imperial.ac.uk},\cr Gianluca Campanella
  \email{g.campanella11@imperial.ac.uk} }
  \references{
  Bottolo L, Chadeau-Hyam M ; et al. GUESS-ing polygenic associations
  with multiple phenotypes using a GPU-based Evolutionary Stochastic
  Search algorithm. PLoS Genetics. 2013;9(8):e1003657.

  Bottolo L, Chadeau-Hyam M ; et al. ESS++ : a C++ Object-Oriented
  Algorithm for Bayesian Stochastic Search Model
  Exploration. Bioinformatics. 2011 ; 27 :587-588.

  Bottolo L and Richardson S (2010). Evolutionary
  Stochastic Search for Bayesian model exploration.
  Bayesian Analysis 5(3), 583-618.

  Petretto E, Bottolo L, Langley SR, Heinig M,
  McDermott-Roe C, Sarwar R, Pravenec M, Hubner N, Aitman
  TJ, Cook SA and Richardson S (2010). New insights into
  the genetic control of gene expression using a Bayesian
  multi-tissue approach. PLoS Comput. Biol., 6(4),
  e1000737.
  }
\seealso{
  \code{\link{R2GUESS}}, \code{\link{as.ESS.object}},
  \code{\link{plotMPPI}}, \code{\link{plot.ESS}},
  \code{\link{print.ESS}}, \code{\link{summary.ESS}}
}
\keyword{package}

