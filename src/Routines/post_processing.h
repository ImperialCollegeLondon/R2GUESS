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

#ifndef POST_PROCESSING_H
#define POST_PROCESSING_H

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>

#include "../Classes/Double_Matrices.h"
#include "../Classes/Double_Matrices_cont.h"
#include "../Classes/Prior_param.h"
#include "../Classes/String_Matrices.h"
#include "../Routines/cond_post.h"
#include "../Routines/matrix_handling.h"

using namespace std; 

int getUniqueList(vector<vector<unsigned int> > &uniqueListModels,
                  const vector<vector<unsigned int> >& listModels,
                  const vector<unsigned int>& n_modelsVisited,
                  const unsigned int& burnIn,
                  const unsigned int& pX,
                  const unsigned int& nConfounders);

void getLogPost(vector<vector<double> >& margLogPostVec,
                  vector<vector<unsigned int> >& uniquelistModels,
                  gsl_matrix *matX,
                  gsl_matrix *matY,
                  Prior_param PR,
                  bool gPriorFlag,
                  bool indepPriorFlag,
                  bool gSampleFlag,
                  double lambda,
                  double gMean,
                  bool cudaFlag);

void getAndSortPostGam(gsl_vector* postGamVec,
                               gsl_permutation* idxPostGamSort,
                               const vector<vector<double> >& margLogPostVec);

double combineAndPrintBestModel(ofstream& fOut,
                              gsl_permutation* const idxPostGamSort,
                              gsl_vector* const postGamVec,
                              const vector<vector<double> >& margLogPostVec,
                              const vector<vector<unsigned int> >& uniqueListModels,
                              const unsigned int& nRetainedModels,
                              const unsigned int& posNullModel,
                              const bool& fullOutput,
                              const unsigned int& nConfounders);


void getAndPrintMargGam(ofstream& fOut,
			    const vector<vector<unsigned int> >& uniqueListModels,
			    gsl_vector* const postGamVec,
			    const unsigned int& pX,
			    const unsigned int& nConfounders);

#endif
