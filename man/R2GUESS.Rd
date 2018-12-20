\name{R2GUESS}
\alias{R2GUESS}
\title{Wrapper function that reads the input files and parameter values required by
  GUESS, runs the C++ code from R and stores the main GUESS output in
  an \code{ESS} object}
\usage{
  R2GUESS(dataY, dataX, path.input, path.output, path.par,
    path.init = NULL, file.par, file.init = NULL,
    file.log = NULL, nsweep, burn.in, Egam,
    Sgam, root.file.output, time = TRUE, top = 100,
    history = TRUE, label.X = NULL, label.Y = NULL,
    choice.Y = NULL, nb.chain, conf = NULL, cuda = TRUE,
    MAP.file = NULL, time.limit=NULL,seed=NULL)
}
\arguments{
\item{dataY}{either a one element character vector (such as
    '\code{dataY.txt}') or a data frame. If \code{dataY} is entered as
    a character vector, it specifies, assuming that data are in the
    \code{path.input} folder, the location of the response
    matrix. In the corresponding file observations are presented in
    rows, and the (possibly multivariate) outcome(s) in columns. The
    first two rows (single integers) represent the number of rows
    (\code{n}) and columns (\code{q}) in the matrix. If a data frame
    argument is passed, it links to a \code{nxq} numerical matrix
    compiling the observed responses.}

\item{dataX}{either a one element character vector (such as
    '\code{dataX.txt}') or a data frame. If \code{dataX} is entered as
    a character vector, it specifies, assuming that data are in the
    \code{path.input} folder, the location of the predictor
    matrix. In the corresponding file observations are presented in
    rows, and the predictors in columns. The first two rows (single
    integers) represent the number of rows (\code{n}) and columns
    (\code{p}) in the matrix. If a data frame argument is passed, it
    links to a \code{nxq} numerical matrix compiling the observed
    predictors.}

\item{path.input}{path linking to the directory containing the data
    (\code{dataX} and \code{dataY}). If
    \code{dataX} or/and \code{dataY} have
    been entered as data frame(s), the function will generate the
    corresponding text files required to run \code{GUESS} in the
    \code{path.input} folder.}

  \item{path.output}{path indicating the directory in which output
      files will be saved.}

  \item{path.par}{path indicating the directory in which to find the
      parameter file needed to run \code{GUESS}.}

  \item{path.init}{path indicating the location of the file describing the
      initial guess of the MCMC procedure (i.e. the variables to include in the initial
      model).}

  \item{file.par}{name of the parameter file containing all
      user-specified parameters required to set up the run and the
      features of the moves. This file is located in \code{path.par}
      and contains fields that are extensively described in
      \url{http://www.bgx.org.uk/software/GUESS_Doc_short.pdf}. These
      parameters are not mandatory and, if not specified, they will be
      set to their default values, also given in documentation.  An
      example of this file is provided in the package.}

  \item{file.init}{name of the file specifying which
  variables to include at the first iteration of the MCMC
  run. The first row of the file is a single scalar
  representing the number of rows (# variables to include).
  Subsequent rows indicate the position of the covariates
  to include.  This file is optional and if not specified
  (\code{default=NULL}), initial guesses of the MCMC algorithm
  will be derived from a step-wise regression approach.}

\item{file.log}{name of the log file. This file compiles in real time
    summary information describing the initial parameters, the
    computational time and state of the run. This file will also
    contain information about moves sampled at each sweep. By default
    (=\code{NULL}), the name is given by the argument
    \code{root.file.output} extended by \code{'_log'} and for
    computational efficiency (especially when GPU is enabled), a
    minimal amount of information is returned.}

  \item{nsweep}{integer specifying the number of sweeps for
  the MCMC run (including the burn-in).}

  \item{burn.in}{integer specifying the number of sweeps to
  be discarded to account the burn-in.}

 \item{Egam}{numeric representing the 'a priori' average
  model size.}

  \item{Sgam}{numeric representing the 'a priori' standard
  deviation of the model size.}


\item{root.file.output}{name specifying the file stem for writing the
    output files in the directory specified by
    \code{path.output}.}

  \item{time}{Boolean value. When \code{time=TRUE} (default value)
  a file recording the time each sweep took will be
  created and saved in  \code{path.output} directory.}

\item{top}{number of top models to be reported in the output. The
    default value is 100.}

\item{history}{Boolean value. When \code{history=TRUE} (default
    value), a number of additional output files that record the
    history of each move is provided. See section 5 of
    \url{http://www.bgx.org.uk/software/GUESS_Doc_short.pdf} for more
    details.}

\item{label.X}{a character vector specifying the name of the
    predictors. If not specified (=NULL), variables are labelled by
    their position in the matrix. Predictors name and information is
    given in the \code{\link{MAP.file}} in the case of SNP data (field
    \code{SNPName}).}

  \item{label.Y}{a character vector specifying the name of
  the outcomes. If not specified (=NULL), the outcomes are
  labelled Y1,..Yq, where q is the number of columns in the
  outcome matrix or will be named after the argument
  \code{dataY} (if specified by a data frame).}

\item{choice.Y}{a character vector or a numeric vector specifying
    which phenotypes in the response matrix \code{dataY} to analyse
    in a joint model. By default, all phenotypes in the response
    matrix will be considered.}

  \item{nb.chain}{an integer specifying the number of
  chains to consider in the evolutionary procedure.}

\item{conf}{either a one element character vector (such as
    '\code{conf.txt}') or a data frame. If \code{conf} is entered as a
    character vector, it specifies, assuming that data are in the
    \code{path.input} folder, the location of the confounder
    matrix. In the corresponding file observations are presented in
    rows, and the values for the confounders in columns. The first two
    rows (single integers) represent the number of rows (\code{n}) and
    columns (\code{k}) in the matrix. If a data frame argument is
    passed, it links to a \code{nxk} numerical matrix compiling the
    observed confounders. If specified, the function will substitute
    the response matrix by the residuals from the linear model
    regressing the confounders against the outcomes.}

\item{cuda}{a boolean value. cuda=TRUE redirects linear algebra
    operations towards the GPU. On non-CULA compatible platforms, this
    option will be ignored.}

  \item{MAP.file}{either a one element character vector or a data
      frame. If a character vector is used, it specifies, assuming that data are in the
    \code{path.input} folder, the location of the annotation
    file. In the corresponding file, predictors are presented in
    rows, and are described as a \code{\link{MAP.file}}. If a data frame
    argument is passed, it links to a \code{px3} matrix.
}

\item{time.limit}{ a numerical value specifying the maximum computing
    time (in hours) for the run. If the run exceeds that value,
    modelling options, parameters value, state of the pseudo random
    number generator, and state of each chain will be saved to enable
    to resume the run exactly at the same point it was interrupted
    (using \code{resume} option). By default (=\code{NULL}) the run
    will go on until its completion.}

\item{seed}{ a integer specifying the random seed used to initialize the
    pseudo-random number generator. If not specified, the seed will
    be initialised using the CPU clock.}}

\value{
  An object of class \code{ESS} containing information listed in
  \code{\link{as.ESS.object}}. The object can subsequently be used to post-process the results using
  provided \code{R} functions (such as \code{\link{summary.ESS}},
  \code{\link{plotMPPI}}, \code{\link{plot.ESS}}).  }


\description{The \code{R2GUESS} function reads and compiles data,
  input files and parameters that are required to run \code{GUESS}
  source code. It automatically runs \code{GUESS} (enabling or not the
  GPU capacity), saves the results and summary files in text
  files. For portability, \code{\link{R2GUESS}} generates an
  \code{ESS} object which compiles information about the input and
  parameters used to run \code{GUESS}, and outputs as detailed in
  \code{\link{as.ESS.object}}.}



\details{ For any of the \code{dataX, dataY} parameters, if a data
  frame argument is passed, a text file labelled
  \code{data-*-C-CODE.txt} will be created in the \code{path.input}
  directory. If \code{conf} is specified, and additional files
  representing the adjusted responses will be created according to the
  file labelling system.This file will be formatted to have the
  suitable structure to be read by the C++ code: individuals presented
  in rows, and observations in columns, with the first two rows
  indicating the number of rows and columns in the matrix. The
  returned \code{ESS} object will include all result files produced by
  the code. The number and type of outputs produced depend on the
  running options chosen. A full description of the available
  output can be found in
  \url{http://www.bgx.org.uk/software/GUESS_Doc_short.pdf} }


\examples{
\dontrun{
path.input <- system.file("Input", package="R2GUESS")
path.output <- tempdir()
path.par <- system.file("extdata", package="R2GUESS")
file.par.Hopx <- "Par_file_example_Hopx.xml"
#you can have a look of the parameter file in
print(paste(path.par,file.par.Hopx,sep=""))
##To reach convergence you may need to increase nsweep=110000 and the burn.in=10000
## RUNNING is APPROX 5 minutes
root.file.output.Hopx <- "Example-GUESS-Y-Hopx"
label.Y <- c("ADR","Fat","Heart","Kidney")
data(data.Y.Hopx)
data(data.X)
data(MAP.file)

modelY_Hopx<-R2GUESS(dataY=data.Y.Hopx,dataX=data.X,choice.Y=1:4,
label.Y=label.Y,,MAP.file=MAP.file,file.par=file.par.Hopx,file.init=NULL,
file.log=NULL,root.file.output=root.file.output.Hopx,path.input=path.input,
path.output=path.output,path.par=path.par,path.init=NULL,nsweep=11000,
burn.in=1000,Egam=5,Sgam=5,top=100,history=TRUE,time=TRUE,
nb.chain=3,conf=NULL,cuda=FALSE)

summary(modelY_Hopx,20) # 20 best models

print(modelY_Hopx)
}
}
\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}
\seealso{
{\code{\link{as.ESS.object}, \link{summary.ESS},\link{as.ESS.object}, \link{plotMPPI}, \link{plot.ESS}}}}


