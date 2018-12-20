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

#ifndef TEMPERATURES_H
#define TEMPERATURES_H 1

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_vector.h>

using namespace std;

class Temperatures
{
public:
  Temperatures();
  ~Temperatures(){};
  unsigned int C;
  double b_t;
  double a_t_den;
  vector < double > a_t;
  vector < double > t;
  unsigned int nbatch;
  vector < double > M;
  double delta_n;
  double optimal;
  unsigned int c_idx;

  void display_Temp_param();

  void set_Temp_param(unsigned int nb_chains,
		      unsigned int pX,
		      double b_t_input,
		      double a_t_den_inf_5k,
		      double a_t_den_5_10k,
		      double a_t_den_sup_10k,
		      unsigned int nbatch_input,
		      vector < double > &M_input,
		      unsigned int burn_in,
		      double optimal_input,
		      bool iso_T_Flag);

};

#endif /* !defined TEMPERATURES_H */
