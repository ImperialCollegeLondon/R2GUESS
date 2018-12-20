\name{Postprocess.R2GUESS}
\alias{Postprocess.R2GUESS}
\title{Performs posterior inference from an interrupted \code{R2GUESS} run.}
\usage{
  Postprocess.R2GUESS(x)
}
\arguments{
  \item{x}{an object of class \code{ESS}.}
}

\value{
  An object of class \code{ESS} whose fields are detailed in
  \code{\link{as.ESS.object}}.}


\description{ The \code{Postprocess.R2GUESS} function calculates the MPPI, the
  MPP and lists the best models visited based on a previous run which
  has been interrupted due to computing time exceeding the user-defined
  limit.}



\details{The \code{Postprocess.R2GUESS} function calculates the MPPI, the
  MPP and lists the best models visited based on a previous run which
  has been interrupted due to computing time exceeding the user-defined
  limit. This function is used to explore results from a run which has
been interrupted and provides intermediate posterior inference for the
MPPI, MPP, and list of best visited models.}


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

modelY_Hopx<-R2GUESS(dataY=data.Y.Hopx,dataX=data.X,,choice.Y=1:4,
label.Y=label.Y,MAP.file=MAP.file,file.par=file.par.Hopx,
file.init=NULL,file.log=NULL,root.file.output=root.file.output.Hopx,
path.input=path.input,path.output=path.output,path.par=path.par,
path.init=NULL,nsweep=510000,burn.in=10000,Egam=5,Sgam=5,top=100,
history=TRUE,time=TRUE,nb.chain=3,conf=0,cuda=FALSE,time.limit=1)

modelY_Hopx_postprocess <- Postprocess.R2GUESS(modelY_Hopx)

}
}
\author{
Benoit Liquet, \email{b.liquet@uq.edu.au},\cr
Marc Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr
Leonardo Bottolo \email{l.bottolo@imperial.ac.uk},\cr
Gianluca Campanella \email{g.campanella11@imperial.ac.uk}
}
\seealso{
\code{\link{Resume.R2GUESS}, \link{Extend.R2GUESS}, \link{as.ESS.object}}}


