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

#include "rand.h"
#include "struc.h"

using namespace std;

double myrand(gsl_rng *RandomNumberGenerator)
{
   return(gsl_rng_uniform(RandomNumberGenerator));
}

void smyrand(long seed,gsl_rng *RandomNumberGenerator)
{
   gsl_rng_set(RandomNumberGenerator, seed);
}

void readRNG(FILE *fRNG,gsl_rng *RandomNumberGenerator){
  gsl_rng_fread(fRNG,RandomNumberGenerator);
}

void writeRNG(FILE *fRNG,gsl_rng *RandomNumberGenerator){
  gsl_rng_fwrite(fRNG,RandomNumberGenerator);
}

double gennor( double av, double sd,gsl_rng *RandomNumberGenerator)
{
   return( av + gsl_ran_gaussian( RandomNumberGenerator, sd ) );
}

int genBernoulli( double p,gsl_rng *RandomNumberGenerator)
{
   return( gsl_ran_bernoulli( RandomNumberGenerator, p) );
}

unsigned int SampleFromDiscrete_new( vector<double> &cdf,gsl_rng *RandomNumberGenerator)
{
  unsigned int k = 0;
  double u = myrand(RandomNumberGenerator);
  while( u > cdf[k] && k < cdf.size()){
    k++;
  }
  return k;
}

unsigned int SampleFromDiscrete_non_cum(vector<double> &pbty,gsl_rng *RandomNumberGenerator)
{
  unsigned int k = 0;
  vector<double> cdf;
  double u = myrand(RandomNumberGenerator);

  cdf.push_back(pbty[0]);

  for(unsigned int i=1;i<pbty.size();i++){
    cdf.push_back(cdf[i-1]+pbty[i]);
  }

  while( u > cdf[k] && k < cdf.size()){
    k++;
  }
  return k;
}

int SampleFromDiscrete_All_exchange(gsl_matrix *description_all_exchange,gsl_rng *RandomNumberGenerator)
{
  unsigned int k = 0;
  vector<double> cdf;
  double u = myrand(RandomNumberGenerator);
  unsigned int ncol=description_all_exchange->size2;

  double pbty=description_all_exchange->data[2*ncol];
  cdf.push_back(pbty);
  for(unsigned int i=1;i<ncol;i++){
    pbty=description_all_exchange->data[2*ncol+i];
    cdf.push_back(cdf[i-1]+pbty);
  }

  while( u > cdf[k] && k < cdf.size()){
    k++;
  }
  cdf.clear();
  return k;
}

void My_Permut_unsigned_int(gsl_permutation *MyPerm,gsl_rng *RandomNumberGenerator)
{
  gsl_ran_shuffle (RandomNumberGenerator, MyPerm->data, MyPerm->size, sizeof(size_t));
}

void My_gsl_ran_choose_double(void * dest, size_t k, void * source, size_t n,gsl_rng *RandomNumberGenerator)
{
  gsl_ran_choose(RandomNumberGenerator, dest, k, source, n, sizeof(double));
}

void My_gsl_ran_choose_u_int(void * dest, size_t k, void * source, size_t n,gsl_rng *RandomNumberGenerator)
{
  gsl_ran_choose(RandomNumberGenerator, dest, k, source, n, sizeof(unsigned int));
}
