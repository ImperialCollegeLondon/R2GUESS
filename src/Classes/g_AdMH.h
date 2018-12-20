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

#ifndef G_AdMH_H
#define G_AdMH_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_vector.h>

#include "../Routines/matrix_handling.h"

using namespace std;

class g_AdMH
{
public:
  g_AdMH();
  ~g_AdMH(){};
  unsigned int  n_batch;
  double optimal;
  double ls;
  double delta_n;

  unsigned int G_tilda_accept;
  unsigned int G_tilda_accept_ins;
  unsigned int G_tilda_n_sweep;
  unsigned int G_tilda_n_sweep_ins;
  vector < double > M;
  vector < double > Ls;
  
  void set_g_AdMH(int g_sample,
		  unsigned int n_batch_from_read,
		  double g_AdMH_optimal_from_read,
		  double g_AdMH_ls_from_read,
		  unsigned int pX,
		  unsigned int burn_in,
		  double g_M_min_input,
		  double g_M_max_input);
  

  void display_g_AdMH();


};

#endif /* !defined G_AdMH_H */
