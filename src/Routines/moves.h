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

#ifndef MOVES_H
#define MOVES_H

#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_cdf.h>
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

#include "../Classes/CM.h"
#include "../Classes/Double_Matrices.h"
#include "../Classes/Double_Matrices_cont.h"
#include "../Classes/DR.h"
#include "../Classes/g_AdMH.h"
#include "../Classes/Int_Matrices.h"
#include "../Classes/Move_monitor.h"
#include "../Classes/Prior_param.h"
#include "../Classes/Temperatures.h"
#include "../Routines/cond_post.h"
#include "../Routines/guess_math.h"
#include "../Routines/matrix_handling.h"
#include "../Routines/rand.h"

void Gibbs_move(Double_Matrices mat_log_marg,
		Double_Matrices mat_log_cond_post,
		gsl_matrix *mat_X,
		gsl_matrix *mat_Y,
		Temperatures *t_tun,
		gsl_permutation *MyPerm,
		vector < vector <unsigned int> > &vect_gam,
		unsigned int sweep,
		bool gPriorFlag,
		bool indepPriorFlag,
		bool gSampleFlag,
		double lambda,
		double g,
		Prior_param PR,
		Move_monitor *My_Move_monitor,
		vector <unsigned int > &chain_idx,
		vector <unsigned int > &n_Models_visited,
                bool cudaFlag,
                unsigned int nConfounders,
                unsigned int maxPX,
                gsl_rng *RandomNumberGenerator);

void intialize_chain_idx(vector <unsigned int > &chain_idx,
			 unsigned int nb_chains);

gsl_matrix *description_exch_moves(unsigned int nb_chains);

void All_exchange_move(gsl_matrix *description_exchange_move,
		       vector <unsigned int > &chain_idx,
		       Double_Matrices mat_log_cond_post,
		       Temperatures *t_tun,
		       unsigned int sweep,
		       Move_monitor *My_Move_monitor,
		       gsl_rng *RandomNumberGenerator);

void DR_move(DR *My_DR,
	     vector <unsigned int > &chain_idx,
	     Double_Matrices mat_log_cond_post,
	     Temperatures *t_tun,
	     unsigned int sweep,
	     Move_monitor *My_Move_monitor,
	     gsl_rng *RandomNumberGenerator);

void sample_g(Double_Matrices mat_log_marg,
	      Double_Matrices mat_log_cond_post,
	      g_AdMH *My_g_AdMH,
	      Temperatures *t_tun,
	      vector < vector <unsigned int> > &vect_gam,
	      gsl_matrix *mat_X,
	      gsl_matrix *mat_Y,
	      bool gPriorFlag,
	      bool indepPriorFlag,
	      bool gSampleFlag,
	      double lambda,
	      double &g,
	      Prior_param PR,
	      unsigned int sweep,
	      Move_monitor *My_Move_monitor,
	      vector <unsigned int > &chain_idx,
	      vector <unsigned int > &n_Models_visited,
              bool cudaFlag,
              gsl_rng *RandomNumberGenerator);



void FSMH_move(Double_Matrices mat_log_marg,
	       Double_Matrices mat_log_cond_post,
	       gsl_matrix *mat_X,
	       gsl_matrix *mat_Y,
	       Temperatures *t_tun,
	       gsl_permutation *MyPerm,
	       vector < vector <unsigned int> > &vect_gam,
	       unsigned int sweep,
	       bool gPriorFlag,
	       bool indepPriorFlag,
	       bool gSampleFlag,
	       double lambda,
	       double g,
	       Prior_param PR,
	       Move_monitor *My_Move_monitor,
	       vector <unsigned int > &chain_idx,
	       vector <unsigned int > &n_Models_visited,
               bool cudaFlag,
               unsigned int nConfounders,
               unsigned int maxPX,
               gsl_rng *RandomNumberGenerator);

void temp_placement(Temperatures *t_tun,
		    DR *My_DR,
		    Move_monitor *My_Move_monitor,
		    unsigned int sweep,
		    unsigned int n_vars_in_last_chain,
		    unsigned int nX,
		    bool iso_T_Flag);

void Crossover_move(Double_Matrices mat_log_marg,
		    Double_Matrices mat_log_cond_post,
		    gsl_matrix *mat_X,
		    gsl_matrix *mat_Y,
		    Temperatures *t_tun,
		    vector < vector <unsigned int> > &vect_gam,
		    bool gPriorFlag,
		    bool indepPriorFlag,
		    bool gSampleFlag,
		    double lambda,
		    double g,
		    Prior_param PR,
		    Move_monitor *My_Move_monitor,
		    vector <unsigned int > &chain_idx,
		    unsigned int sweep,
		    CM *My_CM,
		    vector <unsigned int > &n_Models_visited,
                    bool cudaFlag,
                    unsigned int maxPX,
                    gsl_rng *RandomNumberGenerator);

void computeAndSampleCondPostBoltz(vector<double>& condPostBoltzProb,
					    vector<unsigned int>& sampledChains,
					    const unsigned int& nChains,
					    const unsigned int& cTrunc,
					    const unsigned int& sweep,
					    const Double_Matrices& matLogCondPost,
					    const vector<unsigned int>& chainIdx,
					    Temperatures* const tTun,
					    double& chosenT,
					    gsl_rng *RandomNumberGenerator);

unsigned int recombine_chains(vector < unsigned int > &vect_gam_prop_c1,
			      vector < unsigned int > &vect_gam_prop_c2,
			      vector < unsigned int > &sampled_chains,
			      vector <unsigned int > &chain_idx,
			      vector < vector <unsigned int> > &vect_gam,
			      double *pos_bkpts,
			      unsigned int n_bkpts);

void define_haplotype_bkpts(vector < unsigned int > &r_idx,
			    unsigned int pos_crsv,
			    gsl_matrix *mat_X,
			    Prior_param PR);

unsigned int recombine_haplotype(vector < unsigned int > &vect_gam_prop_c1,
				 vector < unsigned int > &vect_gam_prop_c2,
				 vector < unsigned int > &sampled_chains,
				 vector <unsigned int > &chain_idx,
				 vector < vector <unsigned int> > &vect_gam,
				 vector <unsigned int > &r_idx);

void get_log_cond_post_log_marg_prop(gsl_vector *prop_log_marg_condPost,
                                     vector < unsigned int > vect_gam_prop,
                                     gsl_matrix *mat_X,
                                     gsl_matrix *mat_Y,
                                     Prior_param PR,
                                     bool gPriorFlag,
                                     bool indepPriorFlag,
                                     bool gSampleFlag,
                                     double lambda,
                                     double g,
                                     bool cudaFlag);

double computeLogCondPostBoltz(gsl_vector* const logCondPostVec,
		     const double& chosenT,
		     const unsigned int& c1,
		     const unsigned int& c2,
		     const unsigned int& cTrunc);

#endif /*  */
