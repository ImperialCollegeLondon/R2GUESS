\name{Resume.R2GUESS}
\alias{Resume.R2GUESS}
\title{Function resuming an interrupted \code{\link{R2GUESS}} run}
\usage{
  Resume.R2GUESS(x, time.limit=NULL)
}
\arguments{
\item{x}{an \code{ESS} object corresponding to a preliminary
    \code{R2GUESS} run which was interrupted due to
    computing time exceeding the user-specified  time limit.}

\item{time.limit}{ a numerical value specifying the maximum computing
    time (in hours) for the continuation of the run. If the run exceeds
    that value, modelling options, parameters value, state of the
    pseudo random number generator, and state of each chain will be
    saved to enable to resume the run exactly at the same point it was
    interrupted (using \code{resume} option). By default
    (=\code{NULL}) the resuming run will go on until its completion.}}

\value{
If the resuming run completes the run, it will return the complete
\code{ESS} object (compiling information listed in
\code{\link{as.ESS.object}} function).}

\description{The \code{Resume.R2GUESS} function will resume a run
that was interrupted due to computation time exceeding the
user-defined time limit.}



\details{ The \code{Resume.R2GUESS} function will continue a run that
  was interrupted due to computation time exceeding the user-defined
  time limit. It will compile data, input files and parameters that
  are required for running GUESS code. Additional iterations
  required to complete the run will be appended to the existing
  history files, and posterior calculations (MPPI, MPP, and the list
  of best models) will be added to the \code{ESS} object. The function
  will use the same pseudo-random number generator as the one used for
  the initial run, and will initialise it at the state it was at the
  end of the original run.}


\examples{
\dontrun{
## First we are creating a run which has been not finished in 1 hour
path.input <- system.file("Input", package="R2GUESS")
path.output <- tempdir()
path.par <- system.file("extdata", package="R2GUESS")
file.par.Hopx <- "Par_file_example_Hopx.xml"
print(paste(path.par,file.par.Hopx,sep=""))
root.file.output.Hopx <- "Example-GUESS-Y-Hopx"
label.Y <- c("ADR","Fat","Heart","Kidney")
data(data.Y.Hopx)
data(data.X)
data(MAP.file)

modelY_Hopx<-R2GUESS(dataY=data.Y.Hopx,dataX=data.X,choice.Y=1:4,
label.Y=label.Y,,MAP.file=MAP.file,file.par=file.par.Hopx,
file.init=NULL,file.log=NULL,root.file.output=root.file.output.Hopx,
path.input=path.input,path.output=path.output,path.par=path.par
,path.init=NULL,nsweep=510000,burn.in=10000,Egam=5,Sgam=5,top=100
,history=TRUE,time=TRUE,nb.chain=3,conf=0,cuda=FALSE,time.limit=1)

modelY_Hopx_resume <- Resume.R2GUESS(modelY_Hopx)

}
}
\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}
\seealso{
{\code{\link{Extend.R2GUESS}, \link{Postprocess.R2GUESS}, \link{as.ESS.object}}}}


