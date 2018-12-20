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

#include "cond_post.h"
#include "struc.h"
#include "rand.h"
#include <cstdlib>

#if _CUDA_
#include <cula.h>
#endif

#define DEBUG 0

using namespace std;

void get_vect_gam_init(vector < vector <unsigned int> > &vect_gam,
		       vector < vector <unsigned int> > &Gam_step_regr,
		       bool iso_T_Flag,
		       unsigned int maxPGamma,
		       gsl_rng *RandomNumberGenerator)
{

  unsigned int n_vars_in=0;

  if(!iso_T_Flag){
    for(unsigned int chain=0;chain<vect_gam.size();chain++){
      for(unsigned int row=0;row<Gam_step_regr.size();row++){
	for(unsigned int col=0;col<Gam_step_regr[row].size();col++){
	  unsigned int temp_pos=Gam_step_regr[row][col];
	  vect_gam[chain][temp_pos]=1;
	  if(chain==0){
	    n_vars_in++;
	  }
	} 
      }
    }
  }
  else{
    unsigned int *source_list= new unsigned int [vect_gam[0].size()];
    for(unsigned int row=0;row<Gam_step_regr.size();row++){
      for(unsigned int col=0;col<Gam_step_regr[row].size();col++){
	unsigned int temp_pos=Gam_step_regr[row][col];
	vect_gam[0][temp_pos]=1;
        n_vars_in++;
        source_list[temp_pos]=temp_pos;
      }
    }
    //Filling the other chains randomly
    //Case 1: No vars in: sample one var for chains #2 to C
    if(n_vars_in==0){
      for(unsigned int chain=1;chain<vect_gam.size();chain++){
	unsigned int *list_sample = new unsigned int [1];
	My_gsl_ran_choose_u_int(list_sample,
				1,
				source_list,
				vect_gam[0].size(),RandomNumberGenerator);
	unsigned int curr_pos=list_sample[0];
	vect_gam[chain][curr_pos]=1;
	delete [] list_sample;
      }
    } 
    else{//if n_vars_in>0: sample n_vars in for each chain #2 to C
      for(unsigned int chain=1;chain<vect_gam.size();chain++){
	unsigned int *list_sample = new unsigned int [n_vars_in];
	My_gsl_ran_choose_u_int(list_sample,
				n_vars_in,
				source_list,
				vect_gam[0].size(),
				RandomNumberGenerator);
	for(unsigned int curr_var=0;curr_var<n_vars_in;curr_var++){
	  unsigned int curr_pos=list_sample[curr_var];
	  vect_gam[chain][curr_pos]=1;
	}
	delete [] list_sample;
      }
    }
    delete [] source_list;
  }

  // Now check that there are not too many variables in and if so,
  // randomly remove some
  if(n_vars_in>maxPGamma){
    unsigned int *toRemove = new unsigned int[n_vars_in-maxPGamma];
    unsigned int *varIn = new unsigned int[n_vars_in];
    for(unsigned int i=0;i<n_vars_in;i++){
      varIn[i]=i;
    }
    gsl_ran_choose(RandomNumberGenerator,toRemove,n_vars_in-maxPGamma,varIn,n_vars_in,sizeof(unsigned int));

    for(unsigned int chain=0;chain<vect_gam.size();chain++){
      unsigned int j=0,k=0;
      for(unsigned int col=0;col<vect_gam[chain].size();col++){
        if(vect_gam[chain][col]==1){
          if(j==toRemove[k]){
            vect_gam[chain][col]=0;
            k++;
          }
          j++;
        }
      }
    }
    delete [] varIn;
    delete [] toRemove;
  }
}

void getEigenDecomposition(gsl_matrix *matXGam,
                           gsl_matrix* gslEigenVecs,
                           gsl_vector* gslEigenVals,
                           unsigned int pXGam){

  gsl_matrix *gslMatrixXGamTXGam=gsl_matrix_calloc(pXGam,pXGam);
  gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,matXGam,matXGam,0.0,gslMatrixXGamTXGam);

  gsl_eigen_symmv_workspace *w=gsl_eigen_symmv_alloc(pXGam);
  gsl_eigen_symmv(gslMatrixXGamTXGam,gslEigenVals,gslEigenVecs,w);
  gsl_eigen_symmv_sort(gslEigenVals,gslEigenVecs,GSL_EIGEN_SORT_VAL_ASC);
  gsl_eigen_symmv_free(w);

}

gsl_matrix *getSGamma(gsl_matrix *matXGam,
                   gsl_matrix *matY,
                   double lambda,
                   double g,
                   gsl_matrix *eigenVecs,
                   gsl_vector *eigenVals,
                   bool gPriorFlag,
                   bool indepPriorFlag,
                   unsigned int pXGam,
                   unsigned int nX,
                   unsigned int pY)
{
  
  unsigned int nQ;
  if(gPriorFlag){
    nQ=nX;
  }else{
    nQ=nX+pXGam;
  }

  gsl_matrix *matSGamma=gsl_matrix_calloc(pY,pY);

  if(pXGam<nQ){
    // Always the case for the powered prior
    gsl_matrix *QSubTYTilde=gsl_matrix_alloc(nQ-pXGam,pY);

    if(pXGam>0){
      gsl_vector *tau=gsl_vector_alloc(pXGam);
      gsl_matrix *matXGamTXGamLambdaOver2=gsl_matrix_calloc(pXGam,pXGam);
      gsl_matrix *matXGamTilde = gsl_matrix_calloc(nQ,pXGam);
      gsl_matrix *matYTilde = gsl_matrix_calloc(nQ,pY);
      if(!gPriorFlag){
        if(indepPriorFlag){
          for(unsigned int j=0;j<pXGam;j++){
            gsl_matrix_set(matXGamTXGamLambdaOver2,j,j,1);
          }
        }else{
          gsl_matrix *matD = gsl_matrix_calloc(pXGam,pXGam);
          gsl_matrix *matDEigenVecs = gsl_matrix_calloc(pXGam,pXGam);
          for(unsigned int j=0;j<pXGam;j++){
            gsl_matrix_set(matD,j,j,pow(gsl_vector_get(eigenVals,j),lambda/2.0));
          }
          gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,matD,eigenVecs,0.0,matDEigenVecs);
          gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,eigenVecs,matDEigenVecs,0.0,
                        matXGamTXGamLambdaOver2);
          gsl_matrix_free(matD);
          gsl_matrix_free(matDEigenVecs);
        }
      }

      gsl_matrix_view viewXGamTilde = gsl_matrix_submatrix(matXGamTilde,0,0,nX,pXGam);
      gsl_matrix_view viewYTilde = gsl_matrix_submatrix(matYTilde,0,0,nX,pY);

      if(gPriorFlag){
        gsl_matrix_memcpy(&(viewXGamTilde.matrix),matXGam);
      }else{
        gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,pow(g,0.5),matXGam,matXGamTXGamLambdaOver2,
                        0.0,&(viewXGamTilde.matrix));
        for(unsigned int j=nX;j<nQ;j++){
          gsl_matrix_set(matXGamTilde,j,j-nX,1);
        }

      }
      gsl_matrix_memcpy(&(viewYTilde.matrix),matY);

      //QR decomposition
      gsl_linalg_QR_decomp(matXGamTilde,tau);
      gsl_matrix *matQTYTilde=gsl_matrix_calloc(nQ,pY);
      gsl_matrix_memcpy(matQTYTilde,matYTilde);
      gsl_linalg_QR_QTmat(matXGamTilde,tau,matQTYTilde);

      //Defining Q_sub
      gsl_matrix_view viewMatQTYTilde = gsl_matrix_submatrix(matQTYTilde,pXGam,0
                                                              ,nQ-pXGam,pY);

      gsl_matrix_memcpy(QSubTYTilde,&(viewMatQTYTilde.matrix));

      gsl_matrix_free(matQTYTilde);
      gsl_matrix_free(matXGamTXGamLambdaOver2);
      gsl_matrix_free(matXGamTilde);
      gsl_matrix_free(matYTilde);
      gsl_vector_free(tau);
    }else{
      gsl_matrix_memcpy(QSubTYTilde,matY);
    }
    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,QSubTYTilde,QSubTYTilde,0.0,matSGamma);

    if(gPriorFlag){
      gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0/(g+1.0),matY,matY,g/(g+1.0),matSGamma);
    }

    gsl_matrix_free(QSubTYTilde);

  }else{
    // This can only be invoked in the g-prior case
    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0/(g+1.0),matY,matY,0,matSGamma);
  }
  
  return matSGamma;

}

#if _CUDA_
void getEigenDecompositionCula(gsl_matrix *matXGam,
                                float* matrixEigenVecs,
                                float* vectorEigenVals,
                                unsigned int pXGam){

  gsl_matrix *gslMatrixXGamTXGam=gsl_matrix_calloc(pXGam,pXGam);
  gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,matXGam,matXGam,0.0,gslMatrixXGamTXGam);

  for(unsigned int j=0;j<pXGam;j++){
    for(unsigned int k=0;k<=j;k++){
      matrixEigenVecs[k*pXGam+j]=gsl_matrix_get(gslMatrixXGamTXGam,j,k);
    }
  }
  culaSsyev('V','L',pXGam,matrixEigenVecs,pXGam,vectorEigenVals);

}

gsl_matrix *getSGammaCula(gsl_matrix *matXGam,
                   gsl_matrix *matY,
                   double lambda,
                   double g,
                   float *matrixEigenVecs,
                   float *vectorEigenVals,
                   bool gPriorFlag,
                   bool indepPriorFlag,
                   unsigned int pXGam,
                   unsigned int nX,
                   unsigned int pY)
{

  //This function is only called if pXGam <= nX (# obs)
  //Otherwise RSP is a pY by pY zero matrix pYxpY

  unsigned int nQ;
  if(gPriorFlag){
    nQ=nX;
  }else{
    nQ=nX+pXGam;
  }

  gsl_matrix *gslMatrixSGamma=gsl_matrix_calloc(pY,pY);

  if(pXGam<nQ){
    // Always the case in powered prior
    gsl_matrix* gslMatrixSubQTYTilde=gsl_matrix_calloc(nQ-pXGam,pY);

    if(pXGam>0){
      float *matrixXGamTXGamLambdaOver2,*matrixXGamTilde,*matrixQTYTilde,*tau;
      matrixXGamTXGamLambdaOver2 = (float*) malloc(pXGam*pXGam*sizeof(float));
      matrixXGamTilde = (float*) malloc(nQ*pXGam*sizeof(float));
      matrixQTYTilde = (float*) malloc(nQ*pY*sizeof(float));
      tau = (float*) malloc(pXGam*sizeof(float));

      if(!gPriorFlag){
        if(indepPriorFlag){
          for(unsigned int i=0;i<pXGam;i++){
            for(unsigned int j=0;j<pXGam;j++){
              matrixXGamTXGamLambdaOver2[j*pXGam+i]=0;
            }
            matrixXGamTXGamLambdaOver2[i*pXGam+i]=1;
          }
        }else{
          for(unsigned int i=0;i<pXGam;i++){
            for(unsigned int j=0;j<pXGam;j++){
              matrixXGamTXGamLambdaOver2[j*pXGam+i]=0.0;
              for(unsigned int k=0;k<pXGam;k++){
                matrixXGamTXGamLambdaOver2[j*pXGam+i]+=matrixEigenVecs[k*pXGam+i]*
                                                      matrixEigenVecs[k*pXGam+j]*
                                                      pow((double)vectorEigenVals[k],lambda/2.0);
              }
            }
          }
        }
      }

      if(gPriorFlag){
        for(unsigned int i=0;i<nQ;i++){
          for(unsigned int j=0;j<pXGam;j++){
            matrixXGamTilde[j*nQ+i]=gsl_matrix_get(matXGam,i,j);
          }
          for(unsigned int j=0;j<pY;j++){
            matrixQTYTilde[j*nQ+i]=gsl_matrix_get(matY,i,j);
          }
        }
      }else{
        for(unsigned int i=0;i<nQ;i++){
          if(i<nX){
            for(unsigned int j=0;j<pXGam;j++){
              matrixXGamTilde[j*nQ+i]=0.0;
              for(unsigned int k=0;k<pXGam;k++){
                matrixXGamTilde[j*nQ+i]+=gsl_matrix_get(matXGam,i,k)*
                                                matrixXGamTXGamLambdaOver2[j*pXGam+k];
              }
              // In the powered case we put the g in here
              // In the g prior case it gets adjusted last of all
              matrixXGamTilde[j*nQ+i]*=pow(g,0.5);
            }
            for(unsigned int j=0;j<pY;j++){
              matrixQTYTilde[j*nQ+i]=gsl_matrix_get(matY,i,j);
            }
          }else{
            for(unsigned int j=0;j<pXGam;j++){
              matrixXGamTilde[j*nQ+i]=0.0;
              if(j==i-nX){
                matrixXGamTilde[j*nQ+i]=1.0;
              }
            }
            for(unsigned int j=0;j<pY;j++){
              matrixQTYTilde[j*nQ+i]=0.0;
            }
          }
        }

      }
      culaSgeqrf(nQ,pXGam,matrixXGamTilde,nQ,tau);

      culaSormqr('L','T',nQ,pY,pXGam,matrixXGamTilde,nQ,tau,matrixQTYTilde,nQ);

      for(unsigned int i=pXGam;i<nQ;i++){
        for(unsigned int j=0;j<pY;j++){
          gsl_matrix_set(gslMatrixSubQTYTilde,i-pXGam,j,matrixQTYTilde[j*nQ+i]);
        }
      }

      free(tau);
      free(matrixXGamTXGamLambdaOver2);
      free(matrixXGamTilde);
      free(matrixQTYTilde);
    }else{
      gsl_matrix_memcpy(gslMatrixSubQTYTilde,matY);
    }

    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,gslMatrixSubQTYTilde,
                                          gslMatrixSubQTYTilde,0.0,gslMatrixSGamma);

    if(gPriorFlag){
      gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0/(g+1.0),matY,matY,
                                          g/(g+1.0),gslMatrixSGamma);
    }

    gsl_matrix_free(gslMatrixSubQTYTilde);

  }else{
    // This can only be invoked in the g prior case when nQ=nX
    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0/(g+1.0),matY,matY,0,gslMatrixSGamma);
  }

  return gslMatrixSGamma;

}

#endif


double getPriorGam(Prior_param PR,
		     unsigned int pX,
		     unsigned int pXGam)
{
  double temp1=gsl_sf_lngamma(PR.a_pi+(double)(pXGam));
  double temp2=gsl_sf_lngamma((double)(pX)+PR.b_pi-(double)(pXGam));
  double temp3=gsl_sf_lngamma((double)(pX)+PR.b_pi+PR.a_pi);
  double res=temp1 + temp2 - temp3;

  return res;

}		   

double getPriorG(Prior_param PR,bool gSampleFlag,double g)
{
  double res=0.0;
  if(gSampleFlag){
    res=invGammaPdf(g,PR.alpha,PR.beta);
  }
  else{
    res=0.0;
  }
  return res;
}

double invGammaPdf(double x,
		   double alpha,
		   double beta)
{

  double temp1 = alpha * log(beta) - gsl_sf_lngamma(alpha);
  double temp2 = -(alpha + 1) * log(x) -beta/x;
  double res=temp1 + temp2;

  return res;

}

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
                    unsigned int pY)
{

  //Building Qk
  double res=0.0;

  gsl_matrix *Qk=gsl_matrix_calloc(pY,pY);
  for(unsigned int row=0;row<pY;row++){
    Qk->data[row*(1+Qk->size2)]=PR.k;
  }

  //Step4: Qk=QK + matSGamma
  gsl_matrix_add(Qk,matSGamma);

  gsl_permutation *p = gsl_permutation_alloc (pY);

  int s=0;
  gsl_linalg_LU_decomp (Qk, p, &s);
  double detQk=gsl_linalg_LU_det (Qk,s);

  gsl_matrix_free(Qk);
  gsl_permutation_free (p);

  double logDetInnerMatrix;
  if(pXGam>0){
    if(!gPriorFlag){
      //Step 5 : Compute |g^(-1)I + (X^TX)^(1+lambda)|
      gsl_matrix *innerMatrix=gsl_matrix_calloc(pXGam,pXGam);
      if(indepPriorFlag){
        for(unsigned int j=0;j<pXGam;j++){
          gsl_matrix_set(innerMatrix,j,j,1.0/g);
        }
        gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,matXGam,matXGam,1.0,innerMatrix);
      }else{
        for(unsigned int j=0;j<pXGam;j++){
          for(unsigned int k=0;k<pXGam;k++){
            double tmp;
            if(j==k){
              tmp=1.0/g;
            }else{
              tmp=0.0;
            }
            for(unsigned int r=0;r<pXGam;r++){
              tmp+=matrixEigenVecs[r*pXGam+j]*matrixEigenVecs[r*pXGam+k]*pow((double)vectorEigenVals[r],1.0+lambda);
            }
            gsl_matrix_set(innerMatrix,j,k,tmp);
          }
        }
      }

      p = gsl_permutation_alloc (pXGam);
      s=0;

      gsl_linalg_LU_decomp (innerMatrix, p, &s);
      logDetInnerMatrix=log(gsl_linalg_LU_det(innerMatrix,s));

      gsl_matrix_free(innerMatrix);
      gsl_permutation_free(p);
    }else{
      logDetInnerMatrix=(double)pXGam*(log(1+g)-log(g));
    }
  }else{
    logDetInnerMatrix=0.0;
  }

  res=-((double)pY/2.0)*(double)(pXGam)*log(g)-
         ((double)pY/2.0)*logDetInnerMatrix-
           ((PR.delta+(double)nX+(double)pY-1.0)/2.0)*log(detQk);

  return res;

}

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
			 bool cudaFlag)
{
 
  gsl_matrix *matSGamma;
  float *matrixEigenVecs,*vectorEigenVals;

  if(pXGam>0){
    matrixEigenVecs = (float*) malloc(pXGam*pXGam*sizeof(float));
    vectorEigenVals = (float*) malloc(pXGam*sizeof(float));

#if _CUDA_
    if(cudaFlag){
      if(!(gPriorFlag||indepPriorFlag)){
        getEigenDecompositionCula(matXGam,matrixEigenVecs,vectorEigenVals,pXGam);
      }
      matSGamma=getSGammaCula(matXGam,matY,lambda,g,matrixEigenVecs,
                                vectorEigenVals,gPriorFlag,indepPriorFlag,pXGam,
                                   nX,pY);
    }else{
#endif

      gsl_matrix *gslEigenVecs = gsl_matrix_calloc(pXGam,pXGam);
      gsl_vector *gslEigenVals = gsl_vector_calloc(pXGam);
      if(!(gPriorFlag||indepPriorFlag)){
        getEigenDecomposition(matXGam,gslEigenVecs,gslEigenVals,pXGam);

        // Here we need to copy across the gsl vectors to the float matrices
        for(unsigned int i=0;i<pXGam;i++){
          vectorEigenVals[i]=gsl_vector_get(gslEigenVals,i);
          for(unsigned int j=0;j<pXGam;j++){
            matrixEigenVecs[j*pXGam+i]=gsl_matrix_get(gslEigenVecs,i,j);
          }
        }
      }
      matSGamma=getSGamma(matXGam,matY,lambda,g,gslEigenVecs,gslEigenVals,
                            gPriorFlag,indepPriorFlag,pXGam,nX,pY);

      gsl_matrix_free(gslEigenVecs);
      gsl_vector_free(gslEigenVals);

#if _CUDA_
    }
#endif

  }else{
    gsl_matrix *gslEigenVecs = NULL;
    gsl_vector *gslEigenVals = NULL;
    matSGamma=getSGamma(matXGam,matY,lambda,g,gslEigenVecs,gslEigenVals,
                          gPriorFlag,indepPriorFlag,pXGam,nX,pY);
  }

  logMargLik=getLogMarg(PR,matXGam,matSGamma,matrixEigenVecs,vectorEigenVals,
                          lambda,g,gPriorFlag,indepPriorFlag,pXGam,nX,pY);

  gsl_matrix_free(matSGamma);
  if(pXGam>0){
    free(matrixEigenVecs);
    free(vectorEigenVals);
  }
  
  if(!GUESS_isfinite(logMargLik)){
    logMargLik=-1E9;
  }

  double logPGam=getPriorGam(PR,pX,pXGam);
  if(!GUESS_isfinite(logPGam)){
    logPGam=-1E9;
  }

  double logPG=getPriorG(PR,gSampleFlag,g);
  if(!GUESS_isfinite(logPG)){
    logPG=-1E9;
  }

  logPosterior=logPGam + logPG + logMargLik;
}

