\name{as.ESS.object}
\alias{as.ESS.object}
\title{Compiles the main input and output files from a previous run of
  \code{R2GUESS} and creates an ESS object.}

\usage{ as.ESS.object(dataY, dataX, path.input, path.output,
  root.file.output, label.X = NULL, label.Y = NULL, path.par,
  path.init = NULL, file.par, file.init = NULL, file.log = NULL,
  MAP.file = NULL, command=TRUE) }



\arguments{
\item{dataY}{a character vector (such as
    '\code{dataY.txt}') specifying, assuming that data are in the
    \code{path.input} folder, the location of the response
    matrix. In the corresponding file observations are presented in
    rows, and the (possibly multivariate) outcome(s) in columns. The
     first two rows (single integers) represent the number of rows
    (\code{n}) and columns (\code{q}) in the matrix.}

\item{dataX}{is a character vector (such as
    '\code{dataX.txt}') specifying, assuming that data are in the
    \code{path.input} folder, the location of the predictor
    matrix. In the corresponding file observations are presented in
    rows, and the predictors in columns. The first two rows (single
    integers) represent the number of rows (\code{n}) and columns
    (\code{p}) in the matrix.}

\item{path.input}{path linking to the directory containing the data
    (\code{dataX} and \code{dataY}). If
    \code{dataX} or/and \code{dataY} have been entered
    as data frame(s), the function will generate the corresponding
    text files required to run \code{GUESS} in the \code{path.input} folder.}

  \item{path.output}{path indicating the directory in which output
      files are saved.}

  \item{root.file.output}{name specifying the file stem of the
      different output files in the \code{path.output} directory.}

  \item{label.X}{a character vector specifying the name of the
      predictors. If not specified (=NULL), the variables are labelled
      by their position in the matrix. Predictors name and information
      can be given in the \code{\link{MAP.file}} in the case of SNP
      data (field \code{SNPName}).}

  \item{label.Y}{a character vector specifying the name of
  the outcomes. If not specified (=NULL), the outcomes are
  labelled by Y1,..Yq where q is the dimension of the
  response matrix, or will be the name of the argument
  \code{dataY} (if specified by a data frame).}

  \item{path.par}{path to the directory containing the
  parameter file (argument \code{file.par})}

  \item{path.init}{path to the directory containing the \code{init}
  file (argument \code{file.init}) specifying which variable
  were included at the first iteration of the MCMC run. By
  default (\code{file.init=NULL}) no init file is
  required.}

\item{file.par}{name of the parameter file containing all the
    user-specified parameters used to set up the run and the features
    of the moves. This file is located in \code{path.par} and contains
    fields that are extensively described in
    \url{http://www.bgx.org.uk/software/GUESS_Doc_short.pdf}.}

\item{file.init}{name of the file specifying which variable have been
    included at the iteration of the MCMC run.}

\item{file.log}{name of the log file. This file compiles in real time
    summary information describing the initial parameters, the
    computational time and state of the run. This file will also
    contains information about moves sampled at each sweep. By default
    (=\code{NULL}), the name is given by the argument
    \code{root.file.output} extended by \code{'_log'} and for
    computational efficiency (especially when GPU is enabled) a
    minimal amount of information is returned.}

  \item{MAP.file}{is either a one element character vector or a data
      frame. If a character vector is used, it specifies, assuming that data are in the
    \code{path.input} folder, the location of the annotation
    file. In the corresponding file each predictor is presented in
    rows, and are described as a \code{\link{MAP.file}}. If a data frame
    argument is passed, it links to a \code{px3} matrix.
}
\item{command}{Boolean specifying whether the automatically
    generated C++ command line is saved in the object or not.}
}

\value{An object of class \code{ESS} which compiles the following
  information:
\item{\code{dataY}}{a character vector defining the location of the response matrix, assuming that data are in the \code{path.input} folder.}
\item{\code{dataX}}{a character vector defining the location of the predictor matrix, assuming that data are in the \code{path.input} folder.}
\item{\code{path.input}}{path linking to the directory containing the data (\code{dataX} and \code{dataY}).}
\item{\code{path.output}}{path indicating the directory in which output files were saved.}
\item{\code{path.par}}{path indicating the directory in which to find
    the parameter file used for the run.}
\item{\code{path.init}}{path indicating the location of file
    describing the initial guess of the MCMC procedure. If no
    \code{init} files were specified, the field is set to \code{NULL}. }
\item{\code{time}}{Boolean value indicating if a file recording the
    time each sweep took has been created and saved in \code{path.output} directory.}
\item{\code{file.par}}{name of the parameter file containing all the
    user-specified parameters used to set up the run and the features of the moves.}
\item{\code{file.init}}{name of the file specifying which variables
    were arbitrarily included at the iteration of the MCMC run. If no \code{init} file was specified (=\code{NULL}),
    initial guesses were defined by a stepwise regression approach.}
\item{\code{file.log}}{location of the log file.}
\item{\code{root.file.output}}{file name specifying the file stem used
    to write the output files in the directory specified by \code{path.output}.}
\item{\code{nsweep}}{integer specifying the number of sweeps of the MCMC run (including the burn-in).}
\item{\code{top}}{the number of top models that are reported in the output.}
\item{\code{BestModels}}{A list describing the best model
    visited, with respect to the fields listed in the \code{\link{summary.ESS}}.}
\item{\code{label.X}}{a character vector specifying the name of the
    predictors. If not specified (=NULL), the variables are labelled
    by their position in the matrix from 1 to p.}
\item{\code{label.Y}}{a character vector specifying the name of the
    outcomes. If not specified (=NULL), the outcomes are labelled by
    Y1,..Yq, where q is the dimension of the outcome matrix.}
\item{\code{p}}{the number of predictors in the X matrix.}
\item{\code{q}}{the number of outcomes in the response matrix.}
\item{\code{n}}{the number of observations.}
\item{\code{nb.chain}}{the number of chains in the evolutionary algorithm.}
\item{\code{burn.in}}{integer specifying the number of sweeps which
    were discarded to account for burn-in.}
\item{\code{conf}}{a character vector defining the location of the
    file compiling observed values for the confounders of interest.}
\item{\code{cuda}}{a boolean value indicating if linear algebra
    operations have been re-routed towards the GPU.}
\item{\code{Egam}}{a priori average model size.}
\item{\code{Sgam}}{a priori standard
  deviation of the model size.}
\item{\code{MAP.file}}{a character vector specifying the location of
    the predictor annotation file, assuming that data are in
    \code{path.input}.}
\item{\code{command}}{a character vector describing the \code{C++} command line
    used to generate the results, if saved.}
\item{\code{seed}}{the random seed used to initialise the
    pseudo-random number generator.}
\item{\code{Finish}}{a Boolean value indicating if the run terminated,
  or was interrupted before reaching the user-defined time limit.}
}


\description{
  The \code{as.ESS.object} function compiles the main information
  relating to a previous run of \code{R2GUESS} and compiles them into
  an \code{ESS} object to be further analyzed. Main parameters (e.g. \code{nsweep, burn.in}) are read from the
  '\code{feature}' file automatically generated at the end of every
  \code{R2GUESS} run. Main outputs are also included in the object to
  enable post-processing and further analyses.
}
\examples{
dataX <- "data-X-C-CODE.txt"
dataY <- "data-Y-ALL-C-CODE.txt"

path.input <- system.file("Input", package="R2GUESS")
path.output <- tempdir()
file.copy(system.file("Output", package="R2GUESS"), path.output, recursive = TRUE)
path.output <- file.path(path.output, "Output")
path.par <- system.file("extdata", package="R2GUESS")
file.par <- "Par_file_example_Hopx.xml"
root.file.output <- "Example-GUESS-Y-Hopx"
label.Y <- c("ADR","Fat","Heart","Kidney")
my.env <- new.env()
data(MAP.file,envir=my.env)
MAP.file <- my.env$MAP.file
modelY_Hopx <-as.ESS.object(dataY=dataY,dataX=dataX,path.input=path.input,
    path.output=path.output,root.file.output=root.file.output,label.X=NULL,
    label.Y=label.Y,path.par=path.par,file.par=file.par,MAP.file=MAP.file)

print(modelY_Hopx)
class(modelY_Hopx)
}
\author{
  Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
  Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}

