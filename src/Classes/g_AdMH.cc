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

#include "g_AdMH.h"
#define DEBUG 0

using namespace std;

g_AdMH::g_AdMH()
{
  n_batch=0;
  optimal=0.0;
  ls=0.0;
  delta_n=0.0;
  G_tilda_accept=0.0;
  G_tilda_accept_ins=0.0;
  G_tilda_n_sweep=0;
  G_tilda_n_sweep_ins=0;
  M.resize(2);

}
  
void g_AdMH::set_g_AdMH(int g_sample,
			unsigned int n_batch_from_read,
			double g_AdMH_optimal_from_read,
			double g_AdMH_ls_from_read,
			unsigned int pX,
			unsigned int burn_in,
			double g_M_min_input,
			double g_M_max_input)
{
  if(g_sample==1){
    n_batch= n_batch_from_read;
    optimal=g_AdMH_optimal_from_read;
    ls=g_AdMH_ls_from_read;
    if(fabs(g_M_min_input-0)<1e-10){
      M[0]=-0.5*log((double) pX);
    }
    else{
      M[0]=g_M_min_input;
    }
    if(fabs(g_M_max_input-0)<1e-10){
      M[1]=0.5*log((double) pX);
    }
    else{
      M[1]=g_M_max_input;
    }
    double temp1=fabs(M[0]-ls);
    double temp2=fabs(M[1]-ls);
    delta_n=max(temp1,temp2);
    delta_n/=(double)(burn_in)/(double)(n_batch);
 
  }
}

void g_AdMH::display_g_AdMH()
{

  cout << endl << "**********************************************************" << endl
       << "******************** g_AdMH parameters ********************" << endl 
       << "\tn_batch = " << n_batch << endl
       << "\toptimal = " << optimal << endl
       << "\tls = " << ls << endl
       << "\tM[0] = " << M[0] << " -- " << "M[1] = " << M[1] << endl
       << "\tdelta_n = " << delta_n << endl
       << "\tG_tilda_accept = " << G_tilda_accept << endl
       << "\tG_tilda_accept_ins = " << G_tilda_accept_ins << endl
       << "\tG_tilda_n_sweep = " << G_tilda_n_sweep << endl
       << "\tG_tilda_n_sweep_ins = " << G_tilda_n_sweep_ins << endl
       << "\tLs " << endl << "\t";
  for(unsigned int col=0;col<Ls.size();col++){
    cout << Ls[col] << " ";
  }
  cout << endl
       << "**********************************************************" << endl
       << "**********************************************************" << endl << endl;
  
}
