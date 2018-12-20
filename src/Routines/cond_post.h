/* This file is part of GUESS.
 *      Copyright (c) Marc Chadeau-Hyam (m.chadeau@imperial.ac.uk)
 *                    Leonardo Bottolo (l.bottolo@imperial.ac.uk)
 *                    David Hastie (d.hastie@imperial.ac.uk)
 *      2010
 *
 * GUESS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GUESS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GUESS.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef COND_POST_H
#define COND_POST_H

#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_vector.h>

#include "../Classes/Double_Matrices_cont.h"
#include "../Classes/Double_Matrices.h"
#include "../Classes/Int_Matrices.h"
#include "../Classes/Prior_param.h"
#include "../Routines/guess_math.h"
#include "../Routines/matrix_handling.h"
#include "../Routines/rand.h"

void get_vect_gam_init(vector < vector <unsigned int> > &vect_gam,
		       vector < vector <unsigned int> > &Gam_step_regr,
		       bool iso_T_Flag,
		       unsigned int maxPGamma,
		       gsl_rng *RandomNumberGenerator);

void getEigenDecomposition(gsl_matrix *matXGam,
                           gsl_matrix* gslEigenVecs,
                           gsl_vector* gslEigenVals,
                           unsigned int pXGam);

gsl_matrix *getSGamma(gsl_matrix *mat_X,
                   gsl_matrix *mat_Y,
                   double lambda,
                   double g,
                   gsl_matrix *gslEigenVecs,
                   gsl_vector *gslEigenVals,
                   bool gPriorFlag,
                   bool indepPriorFlag,
                   unsigned int pXGam,
                   unsigned int nX,
                   unsigned int pY);

#if _CUDA_
void getEigenDecompositionCula(gsl_matrix *matXGam,
                               float* matrixEigenVecs,
                               float* vectorEigenVals,
                               unsigned int pXGam);

gsl_matrix *getSGammaCula(gsl_matrix *matXGam,
                             gsl_matrix *matY,
                             double lambda,
                             double g,
                             float* eigenVecs,
                             float* eigenVals,
                             bool gPriorFlag,
                             bool indepPriorFlag,
                             unsigned int pXGam,
                             unsigned int nX,
                             unsigned int pY);


#endif

double getPriorGam(Prior_param PR,
                    unsigned int pX,
                    unsigned int pXGam);

double getPriorG(Prior_param PR,
			  bool gSampleFlag,
			  double g);

double invGammaPdf(double x,
			double alpha,
			double beta);

double getLogMarg(Prior_param PR,
                    gsl_matrix *matXGam,
                    gsl_matrix *matSGamma,
                    float* matrixEigenVecs,
                    float* vectorEigenVals,
                    double lambda,
                    double g,
                    bool gPriorFlag,
                    bool indepPriorFlag,
                    unsigned int pXGam,
                    unsigned int nX,
                    unsigned int pY);

void computeLogPosterior(double& logMargLik,
                         double& logPosterior,
                         gsl_matrix *matXGam,
                         gsl_matrix *matY,
                         Prior_param PR,
                         bool gPriorFlag,
                         bool indepPriorFlag,
                         bool gSampleFlag,
                         double lambda,
                         double g,
                         unsigned int pX,
                         unsigned int pXGam,
                         unsigned int nX,
                         unsigned int pY,
                         bool cudaFlag);

#endif /*  */
