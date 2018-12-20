\name{pairwise.correlation}
\alias{pairwise.correlation}
\title{Calculates and plots the pairwise correlation between outcomes}
\usage{
  pairwise.correlation(Y, label.Y = NULL)
}
\arguments{
  \item{Y}{data frame or matrix corresponding to the responses.}

  \item{label.Y}{character vector indicating the label of the
      outcomes. By default (\code{label.Y}=NULL) labels of \code{Y}
      are set to \code{colnames(Y)}.} }

  \value{ A matrix of the pairwise correlation between outcomes.}


\description{ The
    \code{\link{pairwise.correlation}} function plots an image with
    the pairwise correlation between phenotypes and provides the
    corresponding source matrix.  } \examples{ data(data.Y.Hopx)
    res.cor.Y.Hopx <-
    pairwise.correlation(data.Y.Hopx,label.Y=NULL)$matcor } \author{
    Benoit Liquet, \email{b.liquet@uq.edu.au},\cr Marc
    Chadeau-Hyam \email{m.chadeau@imperial.ac.uk},\cr Leonardo Bottolo
    \email{l.bottolo@imperial.ac.uk},\cr Gianluca Campanella
    \email{g.campanella11@imperial.ac.uk} }
