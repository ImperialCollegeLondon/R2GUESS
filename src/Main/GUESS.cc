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
// v1.1 edits:
// -adding -extend option:  to append extra iterations to a finished run
//                          one argumen: the number of extra sweeps.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>

#include "../Classes/CM.h"
#include "../Classes/Double_Matrices.h"
#include "../Classes/Double_Matrices_cont.h"
#include "../Classes/DR.h"
#include "../Classes/g_AdMH.h"
#include "../Classes/Int_Matrices.h"
#include "../Classes/Int_Matrices_var_dim.h"
#include "../Classes/Move_monitor.h"
#include "../Classes/Prior_param.h"
#include "../Classes/String_Matrices.h"
#include "../Classes/Temperatures.h"
#include "../Routines/cond_post.h"
#include "../Routines/dyn_name.h"
#include "../Routines/matrix_handling.h"
#include "../Routines/moves.h"
#include "../Routines/post_processing.h"
#include "../Routines/rand.h"
#include "../Routines/regression.h"
#include "../Routines/struc.h"
#include "../Routines/xml_file_read.h"

#if _CUDA_
#include <cula.h>
#endif


#define DEBUG 0
#define Missing_data -1
using namespace std;

//******************************************************************
//*main
//******************************************************************

int main(int argc, char *  argv[])
{

  time_t beginTime, currTime;
  beginTime = time(NULL);

  std::clock_t startTime = std::clock();
  std::clock_t tmpTime,endTime;
  double setupTime=0.0,mainLoopTime=0.0,postProcessTime=0.0;

  gsl_rng *RandomNumberGenerator = gsl_rng_alloc( gsl_rng_mt19937 );

  char filename_in_mat_X[1000];
  char filename_in_mat_Y[1000];
  char filename_par[1000];
  char filename_init[1000];

  char path_name_out[1000];
  int na=0;
  long MY_SEED=-1;
  unsigned int n_sweeps=0;
  unsigned int burn_in=0;
  unsigned int n_top_models_from_read=100;
  unsigned int nConfounders=0;
  unsigned int nExtSweeps=0;
  // If gSampleFlag we are also sampling g
  int gSampleFlag=true;
  // If gPriorFlag we are using g-priors otherwise powered priors
  // If indepPriorFlag we are using independent priors
  // Even though g prior and independent prior are special cases of powered
  // priors, computations can be done more cheaply in those special cases
  bool gPriorFlag=true;
  bool indepPriorFlag=false;
  double lambda=-1.0;
  double g_init=1.0; 

  bool standardizeFlag=true;
  bool resumeRun=false;
  bool extendRun=false;
  bool HistoryFlag=false;
  bool Time_monitorFlag=false;
  bool X_Flag=false;
  bool Y_Flag=false;
  bool Par_Flag=false;
  bool Out_full_Flag=false;
  bool Log_Flag=false;
  bool iso_T_Flag=false;
  bool cudaFlag=false;
  bool fixInit=false;
  bool postProcessOnly=false;
  bool run_finished=false;

  bool CMD_set_E_gam=false;
  bool CMD_set_S_gam=false;
  bool CMD_set_N_chain=false;

  unsigned int set_E_gam=1;
  unsigned int set_S_gam=1;
  unsigned int set_N_chain=1;

  double timeLimit = -1;

  na++;
  while(na < argc)
    {
      if ( 0 == strcmp(argv[na],"-X") )
	{
	  X_Flag=true;
	  strcpy(filename_in_mat_X,argv[++na]);
	  if (na+1==argc) break;
	  na++;
	}
      else if ( 0 == strcmp(argv[na],"-Y") )
	{
	  Y_Flag=true;
	  strcpy(filename_in_mat_Y,argv[++na]);
	  if (na+1==argc) break;
	  na++;
	}
      else if ( 0 == strcmp(argv[na],"-par") )
	{
	  Par_Flag=true;
	  strcpy(filename_par,argv[++na]);
	  if (na+1==argc) break;
	  na++;
	}
      else if ( 0 == strcmp(argv[na],"-nsweep") )
	{
	  n_sweeps=atoi(argv[++na]);
	  if (na+1==argc) break;
	  na++;
	}
      else if ( 0 == strcmp(argv[na],"-burn_in") )
	{
	  burn_in=atoi(argv[++na]);
	  if (na+1==argc) break;
	  na++;
	}
      else if ( 0 == strcmp(argv[na],"-seed") ){
	MY_SEED=(long)((atoi(argv[++na])));
	if (na+1==argc) break;
	na++;
      }
      else if ( 0 == strcmp(argv[na],"-out") )
	{
	  strcpy(path_name_out,argv[++na]);
	  if (na+1==argc) break;
	  na++;
	}
      else if ( 0 == strcmp(argv[na],"-out_full") )
	{
	  Out_full_Flag=true;
	  strcpy(path_name_out,argv[++na]);
	  if (na+1==argc) break;
	  na++;
	}
      else if ( 0 == strcmp(argv[na],"-history") )
	{
	  HistoryFlag=true;
	  if (na+1==argc) break;
	  na++;
	}
      else if ( 0 == strcmp(argv[na],"-time") )
	{
	  Time_monitorFlag=true;
	  if (na+1==argc) break;
	  na++;
	}
      else if ( 0 == strcmp(argv[na],"-top") )
	{
	  n_top_models_from_read=atoi(argv[++na]);
	  if (na+1==argc) break;
	  na++;
	}   
      else if ( 0 == strcmp(argv[na],"-log") )
	{
	  Log_Flag=true;
	  na++;
	}
      else if ( 0 == strcmp(argv[na],"-isoT") )
	{
	  iso_T_Flag=true;
	  if (na+1==argc) break;
	  na++;
	}
      else if ( 0 == strcmp(argv[na],"-noRescaleX") )
        {
          standardizeFlag=false;
          if (na+1==argc) break;
          na++;
        }
      else if ( 0 == strcmp(argv[na],"-lambda") )
        {
          gPriorFlag=false;
          lambda=(double)(atof(argv[++na]));
          if(fabs(lambda)<0.000000000001){
            indepPriorFlag=true;
          }
          if (na+1==argc) break;
          na++;
        }
      else if ( 0 == strcmp(argv[na],"-g_set") )
	{
	  gSampleFlag=false;
	  g_init=(double)(atof(argv[++na]));
	  if (na+1==argc) break;
	  na++;
	}
      else if ( 0 == strcmp(argv[na],"-init") )
        {
          fixInit = true;
          strcpy(filename_init,argv[++na]);
          if (na+1==argc) break;
          na++;
        }
      else if ( 0 == strcmp(argv[na],"-cuda") )
        {
#if _CUDA_
          cudaFlag=true;

          int nDevices;
          culaStatus culaStat;
          culaStat = culaGetDeviceCount(&nDevices);
          if(culaStat!=culaNoError){
            cout << "Error detecting how many devices" << endl;
            exit(-1);
          }else{
            cout << "Detected " << nDevices << " gpu devices" << endl;
          }

          //culaThreadExit();
          for(int i=0;i<nDevices;i++){
            culaStat=culaSelectDevice(i);
            if(culaStat!=culaNoError){
              cout << culaGetStatusString(culaStat) << endl;
              cout << culaGetErrorInfo() << endl;
              //culaThreadExit();
              continue;
            }

            culaStat=culaInitialize();
            if(culaStat!=culaNoError){
              cout << culaGetStatusString(culaStat) << endl;
              cout << culaGetErrorInfo() << endl;
              culaShutdown();
              //culaThreadExit();
              continue;
            }

            break;
          }

          if(culaStat!=culaNoError){
            cout << "Unable to initialise GPU" << endl;
            culaShutdown();
            exit(1);
          }else{
            int device;
            culaGetExecutingDevice(&device);
            cout << "Successfully initialised GPU" << endl;
            cout << "Using device " << device << endl;
          }

#else
          cudaFlag=false;
#endif
          if (na+1==argc) break;
          na++;

        }
      else if ( 0 == strcmp(argv[na],"-nconf") )
        {// Confounders must be in the first columns of X
          nConfounders=atoi(argv[++na]);
          if (na+1==argc) break;
          na++;
        }
      else if ( 0 == strcmp(argv[na],"-resume") )
        {
        resumeRun=true;
        if (na+1==argc) break;
          na++;
        }
      else if ( 0 == strcmp(argv[na],"-extend") )
        {
        extendRun=true;
        nExtSweeps=atoi(argv[++na]);
	if (na+1==argc) break;
          na++;
        }
      else if ( 0 == strcmp(argv[na],"-postProcess") )
        {
        postProcessOnly=true;
        if (na+1==argc) break;
          na++;
        }
      else if ( 0 == strcmp(argv[na],"-timeLimit") )
        {
        timeLimit=atof(argv[++na]);
        if (na+1==argc) break;
          na++;
        }
      else if ( 0 == strcmp(argv[na],"-Egam") )
        {
	  CMD_set_E_gam=true;
	  set_E_gam=atoi(argv[++na]);
	  if (na+1==argc) break;
          na++;
        }
      else if ( 0 == strcmp(argv[na],"-Sgam") )
        {
	  CMD_set_S_gam=true;
	  set_S_gam=atoi(argv[++na]);
	  if (na+1==argc) break;
          na++;
        }
      else if ( 0 == strcmp(argv[na],"-n_chain") )
        {
	  CMD_set_N_chain=true;
	  set_N_chain=atoi(argv[++na]);
	  if (na+1==argc) break;
          na++;
        }

      else
        {
          cout << "Unknown option: " << argv[na] << endl;
#if _CUDA_
          if(cudaFlag){
	    
	    culaShutdown();
          }
#endif
          exit(1);
        }
    }
  
  if(!X_Flag){
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "   The predictor matrix X has not been specified, RUN STOPPED" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
#if _CUDA_
    if(cudaFlag){
      culaShutdown();
    }
#endif

    exit(1);
  }

  if(!Y_Flag){
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "   The outcome matrix Y has not been specified, RUN STOPPED" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
#if _CUDA_
    if(cudaFlag){
      culaShutdown();
    }
#endif
    exit(-1);
  }

  if(!Par_Flag){
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "   The parameters matrix has not been specified, RUN STOPPED" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
#if _CUDA_
    if(cudaFlag){
      culaShutdown();
    }
#endif
    exit(-1);
  }

  if(postProcessOnly){
    resumeRun=false;
    fixInit=false;
    extendRun=false;
  }

  unsigned int resumeSweep=0;
  double resumeG;
  string Extension_out=".txt";
  string resumeName=Get_stddzed_name_short(path_name_out,
					   "resume",
					   Extension_out);
  fstream f_resume;
  string rngFileName=Get_stddzed_name_short(path_name_out,
					    "random",
					    ".rng");
  FILE *f_rng;
  if(MY_SEED<0){
    MY_SEED=(long)time(0);
  }
  smyrand((long)(MY_SEED),RandomNumberGenerator);

  if(resumeRun||postProcessOnly || extendRun){
    // Get the current sweep
    f_resume.open(resumeName.c_str(),ios::in);
    f_resume >> resumeSweep;
    f_resume >> resumeG;
    // Need to read in the state of random number generator from file
    f_rng=fopen(rngFileName.c_str(),"rb");

    readRNG(f_rng,RandomNumberGenerator);
    fclose(f_rng);
  }

  if(postProcessOnly){
    if(resumeSweep < burn_in){
      cout << "Post processing cannot be applied because current number of sweeps less than burn in -- stopping run" << endl;
#if _CUDA_
      if(cudaFlag){
        culaShutdown();
      }
#endif
      exit(-1);
    }
  }

  //Reading X matrix.

  fstream f_X;
  f_X.open(filename_in_mat_X,ios::in);
  unsigned int nX;
  unsigned int pX;
  f_X >> nX;
  f_X >> pX;

  gsl_matrix *mat_X=gsl_matrix_alloc(nX,pX);

  for(unsigned int i=0;i<nX;i++){
    for(unsigned int j=0;j<pX;j++){
      double tmp;
      f_X >> tmp;
      gsl_matrix_set(mat_X,i,j,tmp);
    }
  }
  f_X.close();

  //Reading Y matrix.
  fstream f_Y;
  f_Y.open(filename_in_mat_Y);

  unsigned int nY;
  unsigned int pY;
  f_Y >> nY;
  f_Y >> pY;

  gsl_matrix *mat_Y=gsl_matrix_alloc(nY,pY);
  for(unsigned int i=0;i<nY;i++){
    for(unsigned int j=0;j<pY;j++){
      double tmp;
      f_Y >> tmp;
      gsl_matrix_set(mat_Y,i,j,tmp);
    }
  }
  f_Y.close();


  /////Testing the number of variables in Y
  if(nX<pY){
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "There are too many outcomes ("<< pY
	 << ") compared to the number of observations (" << nX
	 << "), run stopped" << endl;
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
#if _CUDA_
    if(cudaFlag){
      //cublasShutdown();
      culaShutdown();
    }
#endif
    exit(-1);
  }


  //////////////////////////////////
  //  Running options
  //////////////////////////////////
  
  if(n_sweeps==0 || burn_in==0){
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "The Number of sweeps and/or the burn-in has not been specified" << endl
	 << "Use -iter and/or -burn-in option(s) in the command line" << endl
	 << "Run stopped" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
#if _CUDA_
    if(cudaFlag){
      //cublasShutdown();
      culaShutdown();
    }
#endif
    exit(-1);
  }
  if(n_sweeps <= burn_in){
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "The Number of sweeps: " << n_sweeps << " is lower than " << endl
	 << "(or equal to) the burn-in: " << burn_in << " -- Run stopped" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl
	 << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
#if _CUDA_
    if(cudaFlag){
      //cublasShutdown();
      culaShutdown();
    }
#endif
    exit(-1);
  }
   //Setting up modelling parameters.
 
  double E_p_gam_from_read=2;
  double Sd_p_gam_from_read=1;
  double delta_from_read=1;
  unsigned int nb_chains=3;
  unsigned int n_top_models=n_sweeps;
  double g=0;
  if(resumeRun || extendRun){
    g=resumeG;
  }else{
    if(!gSampleFlag){
      g=g_init;
    }else{
      g=pow((double)(pX),2);
    }
  }

  double maxPGammaFactor = 10;
  double P_mutation_from_read=0.5;
  double P_sel_from_read=0.5;
  double P_csvr_r_from_read=0.375;
  double P_DR_from_read=0.5;

  //Reading the parameter File
  FILE *fparameter=NULL;
  char str[256]; // string used to read parameters

  fparameter = fopen(filename_par,"r");
  MaXmlTagRead XML_PAR(fparameter); // assign class
  //Reading elements of the par file
  if(XML_PAR.ReadTag("DELTA", 0, 0, str,256)){
    sscanf(str,"%lf",&delta_from_read);
  }
  if(CMD_set_E_gam==false){
    if(XML_PAR.ReadTag("E_P_GAM", 0, 0, str,256)){
      sscanf(str,"%lf",&E_p_gam_from_read);
    }
  }
  else{
    E_p_gam_from_read=set_E_gam;
  }
  if(CMD_set_S_gam==false){
    if(XML_PAR.ReadTag("SD_P_GAM", 0, 0, str,256)){
      sscanf(str,"%lf",&Sd_p_gam_from_read);
    }
  }
  else{
    Sd_p_gam_from_read=set_S_gam;
  }
  if(XML_PAR.ReadTag("MAX_P_GAM_FACTOR", 0, 0, str,256)){
      sscanf(str,"%lf",&maxPGammaFactor);
  }

  if(CMD_set_N_chain==false){
    if(XML_PAR.ReadTag("NB_CHAINS", 0, 0, str,256)){
      sscanf(str,"%u",&nb_chains);
    }
  }
  else{
    nb_chains=set_N_chain;
  }
  if(XML_PAR.ReadTag("P_MUTATION", 0, 0, str,256)){
    sscanf(str,"%lf",&P_mutation_from_read);
  }
  if(XML_PAR.ReadTag("P_SEL", 0, 0, str,256)){
    sscanf(str,"%lf",&P_sel_from_read);
  }
  if(XML_PAR.ReadTag("P_CSRV_R", 0, 0, str,256)){
    sscanf(str,"%lf",&P_csvr_r_from_read);
  }
  if(XML_PAR.ReadTag("P_DR", 0, 0, str,256)){
    sscanf(str,"%lf",&P_DR_from_read);
  }

  if(n_top_models_from_read!=0){
    n_top_models=n_top_models_from_read;
  }

  // Add in the number of confounders
  E_p_gam_from_read = E_p_gam_from_read + nConfounders;
  // Set the limit on the maximum number of factors
  unsigned int maxPGamma = (unsigned int)(E_p_gam_from_read+Sd_p_gam_from_read*maxPGammaFactor);
  if(maxPGamma>nX){
    maxPGamma=nX;
  }

  //Setting up regression parameters.
  double n_Pvalue_enter=0.01;
  double n_Pvalue_remove=0.01;

  if(XML_PAR.ReadTag("N_P_VALUE_ENTER", 0, 0, str,256)){
    sscanf(str,"%lf",&n_Pvalue_enter);
  }
  if(XML_PAR.ReadTag("N_P_VALUE_REMOVE", 0, 0, str,256)){
    sscanf(str,"%lf",&n_Pvalue_remove);
  } 

  
  //Regression Setting up parameters.
  double Pvalue_enter = 1.0 - pow((1.0 - n_Pvalue_enter),(1.0/(double)(pX)));
  double Pvalue_remove = 1.0 - pow((1.0 - n_Pvalue_remove),(1.0/(double)(pX)));
  

  //Moves Parameters
  //g Adaptative M-H
  unsigned int g_n_batch_from_read=100;
  double g_AdMH_optimal_from_read=0.44;
  double  g_AdMH_ls_from_read=0.0;
  double g_M_min_input=0.0;
  double g_M_max_input=0.0;
  if(XML_PAR.ReadTag("G_N_BATCH", 0, 0, str,256)){
    sscanf(str,"%u",&g_n_batch_from_read);
  }
  if(XML_PAR.ReadTag("G_ADMH_OPTIMAL", 0, 0, str,256)){
    sscanf(str,"%lf",&g_AdMH_optimal_from_read);
  }
  if(XML_PAR.ReadTag("G_ADMH_LS", 0, 0, str,256)){
    sscanf(str,"%lf",&g_AdMH_ls_from_read);
  }
  if(XML_PAR.ReadTag("G_M_MIN", 0, 0, str,256)){
    sscanf(str,"%lf",&g_M_min_input);
  }
  if(XML_PAR.ReadTag("G_M_MAX", 0, 0, str,256)){
    sscanf(str,"%lf",&g_M_max_input);
  }
  //Crossover Move
  unsigned int k_max_from_read=2;
  if(XML_PAR.ReadTag("K_MAX", 0, 0, str,256)){
    sscanf(str,"%u",&k_max_from_read);
  }
  //Gibbs Move
  unsigned int Gibbs_n_batch=500;
  if(XML_PAR.ReadTag("GIBBS_N_BATCH", 0, 0, str,256)){
    sscanf(str,"%u",&Gibbs_n_batch);
  }

  //Temperature Placement
  double b_t_input=2.0;
  double a_t_den_inf_5k=2.0;
  double a_t_den_5_10k=4.0;
  double a_t_den_sup_10k=2.0;
  unsigned int temp_n_batch=50;
  double temp_optimal_input=0.5;
  vector < double > M_input;
  M_input.resize(2);
  double M_min_input=1.0;
  double M_max_input=4.0;

  

  if(XML_PAR.ReadTag("B_T", 0, 0, str,256)){
    sscanf(str,"%lf",&b_t_input);
  }
  if(XML_PAR.ReadTag("A_T_DEN_INF_5K", 0, 0, str,256)){
    sscanf(str,"%lf",&a_t_den_inf_5k);
  }
  if(XML_PAR.ReadTag("A_T_DEN_5_10K", 0, 0, str,256)){
    sscanf(str,"%lf",&a_t_den_5_10k);
  }
  if(XML_PAR.ReadTag("A_T_DEN_SUP_10K", 0, 0, str,256)){
    sscanf(str,"%lf",&a_t_den_sup_10k);
  }
  if(XML_PAR.ReadTag("TEMP_N_BATCH", 0, 0, str,256)){
    sscanf(str,"%u",&temp_n_batch);
  }
  if(XML_PAR.ReadTag("TEMP_OPTIMAL", 0, 0, str,256)){
    sscanf(str,"%lf",&temp_optimal_input);
  }
  if(XML_PAR.ReadTag("M_MIN", 0, 0, str,256)){
    sscanf(str,"%lf",&M_min_input);
  }
  if(XML_PAR.ReadTag("M_MAX", 0, 0, str,256)){
    sscanf(str,"%lf",&M_max_input);
  }
 
  M_input[0]=M_min_input;
  M_input[1]=M_max_input;

  fclose(fparameter);



  cout << "**********************************************************" << endl
       << "***************** Setup options **********************" << endl;
  if(postProcessOnly){
    cout << "Post processing previous run only" << endl;
  }
  else{
    if(resumeRun){
      cout << "Resuming previous run" << endl;
    }
    else if(extendRun){
      cout << "Extending previous run for " << nExtSweeps << " sweeps" << endl;
    }
    else{
      cout << "Random seed " << MY_SEED << endl;
    }
    cout << "nb_chains " << nb_chains << endl
	 << "n_sweeps " << n_sweeps << endl
	 << "n_top_models " << n_top_models << endl
	 << "burn_in " << burn_in << endl
	 << "E_p_gam_input " << E_p_gam_from_read-nConfounders << endl
	 << "delta_input " << delta_from_read << endl
	 << "Sd_p_gam_input " << Sd_p_gam_from_read << endl
	 << "Max_p_gam_factor " << maxPGammaFactor << endl
	 << "Max p_gam " << maxPGamma-nConfounders << endl;
    if(gSampleFlag){
      cout << "Sampling g" << endl;
    }else{
      cout << "Not sampling g" << endl;
      cout << "g " << g << endl;
    }
    if(gPriorFlag){
      cout << "Using g prior" << endl;
    }else{
      if(indepPriorFlag){
        cout << "Using independence prior" << endl;
      }else{
        cout << "Using powered prior with lambda = " << lambda << endl;
      }
    }
    cout << "CUDA " << cudaFlag << endl;
    /*cout << "No. confounders " << nConfounders << endl;*/
  }
  cout << "**********************************************************" << endl
       << "**********************************************************" << endl << endl;

  if(!postProcessOnly&&!resumeRun&&!fixInit&&!extendRun){
    cout << "**********************************************************" << endl
       << "*************** Regression parameters ********************" << endl
       << "n_Pvalue_enter " << n_Pvalue_enter << endl
       << "n_Pvalue_enter " << n_Pvalue_enter << endl
       << "Pvalue_enter stepwise " << Pvalue_enter << endl
       << "Pvalue_remove stepwise " << Pvalue_remove << endl
       << "**********************************************************" << endl
       << "**********************************************************" << endl << endl;
  }

  //Testing PATH-file names for output
  
  Extension_out=".txt";
  string Main_Output_name="output_models_history";
  string OutputName=Get_stddzed_name_short(path_name_out,
					   Main_Output_name,
					   Extension_out);
  fstream f_in;
  ofstream f_out;
  ostringstream strStrOut;

  string OutputName_n_vars_in=Get_stddzed_name_short(path_name_out,
						     "output_model_size_history",
						     Extension_out);
  ofstream f_out_n_vars_in;
  ostringstream strStrOutNVarsIn;

  string OutputName_n_models_visited=Get_stddzed_name_short(path_name_out,
							    "output_n_models_visited_history",
							    Extension_out);
  fstream f_in_n_models_visited;
  ofstream f_out_n_models_visited;
  ostringstream strStrOutNModelsVisited;

  string OutputName_log_cond_post_per_chain=Get_stddzed_name_short(path_name_out,
								   "output_log_cond_post_prob_history",
								   Extension_out);
  ofstream f_out_log_cond_post_per_chain;
  ostringstream strStrOutLogCondPostPerChain;

  string OutputName_FSMH=Get_stddzed_name_short(path_name_out,
						"output_fast_scan_history",
						Extension_out);

  ofstream f_out_FSMH;
  ostringstream strStrOutFSMH;

  string OutputName_CM=Get_stddzed_name_short(path_name_out,
					      "output_cross_over_history",
					      Extension_out);

  ofstream f_out_CM;
  ostringstream strStrOutCM;

  string OutputName_AE=Get_stddzed_name_short(path_name_out,
					      "output_all_exchange_history",
					      Extension_out);

  ofstream f_out_AE;
  ostringstream strStrOutAE;

  string OutputName_DR=Get_stddzed_name_short(path_name_out,
					      "output_delayed_rejection_history",
					      Extension_out);
  ofstream f_out_DR;
  ostringstream strStrOutDR;

  string OutputName_g=Get_stddzed_name_short(path_name_out,
					     "output_g_history",
					     Extension_out);
  ofstream f_out_g;
  ostringstream strStrOutG;

  string OutputName_g_adapt=Get_stddzed_name_short(path_name_out,
						   "output_g_adaptation_history",
						   Extension_out);
  ofstream f_out_g_adapt;
  ostringstream strStrOutGAdapt;

  string OutputName_Gibbs=Get_stddzed_name_short(path_name_out,
						 "output_gibbs_history",
						 Extension_out);
  ofstream f_out_Gibbs;
  ostringstream strStrOutGibbs;

  string OutputName_t_tun=Get_stddzed_name_short(path_name_out,
						 "output_temperature_history",
						 Extension_out);
  ofstream f_out_t_tun;
  ostringstream strStrOutTTun;

  vector<vector<unsigned int> > pastModels;
  vector<unsigned int> pastNModelsVisited;

  ios_base::openmode fileMode;
  if(resumeRun || extendRun){
    fileMode=ios::app;
  }else{
    fileMode=ios::out;
  }

  if(HistoryFlag){
    if(resumeRun||postProcessOnly||extendRun){
      pastModels.resize(resumeSweep+1);
      f_in.open(OutputName.c_str(),ios::in);
      if(f_in.fail()){
        cout << "Trying to resume a run where no history was written -- stopping run" << endl;
#if _CUDA_
        if(cudaFlag){
          //cublasShutdown();
          culaShutdown();
        }
#endif
        exit(-1);
      }
      // Remove the first line
      string strtmp;
      for(unsigned int i=0;i<5;i++){
        f_in >> strtmp;
      }
      for(unsigned int i=0;i<resumeSweep+1;i++){
        unsigned int tmp,modelSize;
        double tmp1;
        f_in >> tmp;
        f_in >> modelSize;
        f_in >> tmp1;
        f_in >> tmp1;
        pastModels[i].resize(modelSize+1);
        pastModels[i][0]=modelSize;
        for(unsigned int j=0;j<modelSize;j++){
          f_in >> tmp;
          pastModels[i][j+1]=tmp-1;
        }
      }
      f_in.close();
      if(resumeRun || extendRun){
        f_out.open(OutputName.c_str(),fileMode);
      }
    }
    else{
      f_out.open(OutputName.c_str(),fileMode);
    }
    if(!postProcessOnly){
      if(f_out.fail()){
        cout << "Invalid Path and/or permission rights for " << OutputName << " -- run stopped." << endl;
#if _CUDA_
        if(cudaFlag){
          //cublasShutdown();
          culaShutdown();
        }
#endif
        exit(-1);
      }
      else{
        if(!resumeRun && !extendRun){
          f_out << "Sweep\tModel_size\tlog_marg\tlog_cond_post\tModel"<<endl;
        }
      }
    }

    if(!postProcessOnly){
      f_out_n_vars_in.open(OutputName_n_vars_in.c_str(),fileMode);
      if(f_out_n_vars_in.fail()){
        cout << "Invalid Path and/or permission rights for " << OutputName_n_vars_in << " -- run stopped." << endl;
#if _CUDA_
        if(cudaFlag){
          //cublasShutdown();
          culaShutdown();
        }
#endif
        exit(-1);
      }
      else{
        if(!resumeRun && !extendRun){
          f_out_n_vars_in << "Sweep\t";
          for(unsigned int tmp_chain=0;tmp_chain<nb_chains;tmp_chain++){
            f_out_n_vars_in << "Chain_"<< tmp_chain+1 << "\t";
          }
          f_out_n_vars_in << endl;
        }
      }
    }

    if(resumeRun||postProcessOnly||extendRun){
      pastNModelsVisited.assign(resumeSweep+1,0);
      f_in_n_models_visited.open(OutputName_n_models_visited.c_str(),ios::in);
      if(f_in_n_models_visited.fail()){
        cout << "Trying to resume a run where no history was written -- stopping run" << endl;
#if _CUDA_
        if(cudaFlag){
          //cublasShutdown();
          culaShutdown();
        }
#endif
        exit(-1);
      }
      string tmpstr;
      f_in_n_models_visited >> tmpstr;
      f_in_n_models_visited >> tmpstr;
      for(unsigned int i=1;i<resumeSweep+1;i++){
        unsigned int tmp;
        f_in_n_models_visited >> tmp;
        f_in_n_models_visited >> pastNModelsVisited[i];
      }
      f_in_n_models_visited.close();
      if(resumeRun || extendRun){
        f_out_n_models_visited.open(OutputName_n_models_visited.c_str(),fileMode);
      }
    }
    else{
      f_out_n_models_visited.open(OutputName_n_models_visited.c_str(),fileMode);
    }
    if(!postProcessOnly){
      if(f_out_n_models_visited.fail()){
        cout << "Invalid Path and/or permission rights for " << OutputName_n_models_visited << " -- run stopped." << endl;
#if _CUDA_
        if(cudaFlag){
          //cublasShutdown();
          culaShutdown();
        }
#endif
        exit(-1);
      }
      else{
        if(!resumeRun  && !extendRun){
          f_out_n_models_visited << "Sweep\tn_models_visited" << endl;
        }
      }
    }

    if(!postProcessOnly){
      f_out_log_cond_post_per_chain.open(OutputName_log_cond_post_per_chain.c_str(),fileMode);
      if(f_out_log_cond_post_per_chain.fail()){
        cout << "Invalid Path and/or permission rights for " << OutputName_log_cond_post_per_chain << " -- run stopped." << endl;
#if _CUDA_
        if(cudaFlag){
          //cublasShutdown();
          culaShutdown();
        }
#endif
        exit(-1);
      }
      else{
        if(!resumeRun && !extendRun){
          f_out_log_cond_post_per_chain << "Sweep"<< "\t";
          for(unsigned int tmp_chain=0;tmp_chain<nb_chains;tmp_chain++){
            f_out_log_cond_post_per_chain << "Chain_"<< tmp_chain+1 << "\t";
          }
          f_out_log_cond_post_per_chain << endl;
        }
      }
    }

    if(!postProcessOnly){
      f_out_Gibbs.open(OutputName_Gibbs.c_str(),fileMode);
      f_out_FSMH.open(OutputName_FSMH.c_str(),fileMode);
      f_out_CM.open(OutputName_CM.c_str(),fileMode);
      f_out_AE.open(OutputName_AE.c_str(),fileMode);
      f_out_DR.open(OutputName_DR.c_str(),fileMode);
      if(gSampleFlag){
        f_out_g.open(OutputName_g.c_str(),fileMode);
        f_out_g_adapt.open(OutputName_g_adapt.c_str(),fileMode);
      }
      if(iso_T_Flag==false){
        f_out_t_tun.open(OutputName_t_tun.c_str(),fileMode);
      }
    }
  }else{
    if(resumeRun||postProcessOnly||extendRun){
      cout << "Trying to resume a run where no history was written -- stopping run" << endl;
#if _CUDA_
      if(cudaFlag){
        //cublasShutdown();
        culaShutdown();
      }
#endif
      exit(-1);

    }
  }

  string OutputName_time; 
  ofstream f_out_time;
  ostringstream strStrOutTime;
  if(!postProcessOnly){
    if(Time_monitorFlag){
      OutputName_time=Get_stddzed_name_short(path_name_out,
					     "output_time_monitor",
					     Extension_out);
      f_out_time.open(OutputName_time.c_str(),fileMode);
      if(!resumeRun && !extendRun){
        f_out_time << "Sweep\tTime\tTime_per_eval_model"<<endl;
      }
    }
  }

  string OutputName_best_models=Get_stddzed_name_short(path_name_out,
						       "output_best_visited_models",
						       Extension_out);
  ofstream f_out_best_models;
  f_out_best_models.open(OutputName_best_models.c_str(),ios::out);
  if(f_out_best_models.fail()){
    cout << "Invalid Path and/or permission rights for " << OutputName_best_models << " -- run stopped." << endl;
#if _CUDA_
    if(cudaFlag){
      //cublasShutdown();
      culaShutdown();
    }
#endif
    exit(-1);
  }
  else{
    if(Out_full_Flag){
      f_out_best_models << "Rank\t#Visits\tSweep_1st_visit\t#models_eval_before_1st_visit\tModel_size\tlog_Post_Prob\tModel_Post_Prob\tJeffreys_scale\tModel"<<endl;
    }
    else{
      f_out_best_models << "Rank\t#Visits\tModel_size\tlog_Post_Prob\tModel_Post_Prob\tJeffreys_scale\tModel"<<endl;
    }
  }

  string OutputName_features=Get_stddzed_name_short(path_name_out,
						    "features",
						    Extension_out);
  ofstream f_out_features;
  f_out_features.open(OutputName_features.c_str(),ios::out);
  if(f_out_features.fail()){
    cout << "Invalid Path and/or permission rights for " << OutputName_features << " -- run stopped." << endl;
#if _CUDA_
    if(cudaFlag){
      //cublasShutdown();
      culaShutdown();
    }
#endif
    exit(-1);
  }
  else{
    f_out_features << "Feature\tvalue"<<endl;
  }
  
  
  
  string OutputName_marg_gam=Get_stddzed_name_short(path_name_out,
						    "output_marg_prob_incl",
						    Extension_out);
  ofstream f_out_marg_gam;
  f_out_marg_gam.open(OutputName_marg_gam.c_str(),ios::out);
  if(f_out_marg_gam.fail()){
    cout << "Invalid Path and/or permission rights for " << OutputName_marg_gam << " -- run stopped." << endl;
#if _CUDA_
    if(cudaFlag){
      //cublasShutdown();
      culaShutdown();
    }
#endif
    exit(-1);
  }
  else{
    f_out_marg_gam << "Predictor\tMarg_Prob_Incl"<< endl;
  }

  vector < vector <unsigned int> > Gam_step_regr;
  Gam_step_regr.resize(pY);
  vector<unsigned int>initGam;
  gsl_vector *vect_RMSE = gsl_vector_calloc(pY);
  vector<vector<unsigned int> > resumeGam;
  double resumeCountG;
  double resumeCumG;
  vector<double> resumeT;
  double resumeBT;
  double resumeLS;
  unsigned int resumeDRNCalls;
  unsigned int resumeDRNCallsAdj;
  vector<vector<unsigned int> > resumeDRAccepted;
  vector<vector<unsigned int> > resumeDRProposed;
  vector<unsigned int> resumeChainIndex;

  if(resumeRun||postProcessOnly||extendRun){
    // Read the last state in from file
    f_resume >> resumeCumG;
    f_resume >> resumeCountG;
    for(unsigned int j=0;j<pY;j++){
      f_resume >> vect_RMSE->data[j];
    }
    resumeT.resize(nb_chains);
    for(unsigned int j=0;j<nb_chains;j++){
      f_resume >> resumeT[j];
    }
    f_resume >> resumeBT;
    f_resume >> resumeLS;
    resumeDRAccepted.resize(nb_chains);
    resumeDRProposed.resize(nb_chains);
    f_resume >> resumeDRNCalls;
    f_resume >> resumeDRNCallsAdj;
    for(unsigned int j=0;j<nb_chains;j++){
      resumeDRProposed[j].resize(nb_chains);
      for(unsigned int k=0;k<nb_chains;k++){
        f_resume >> resumeDRProposed[j][k];
      }
    }
    for(unsigned int j=0;j<nb_chains;j++){
      resumeDRAccepted[j].resize(nb_chains);
      for(unsigned int k=0;k<nb_chains;k++){
        f_resume >> resumeDRAccepted[j][k];
      }
    }
    resumeGam.resize(nb_chains);
    resumeChainIndex.resize(nb_chains);
    for(unsigned int j=0;j<nb_chains;j++){
      unsigned int gamSize;
      f_resume >> gamSize;
      f_resume >> resumeChainIndex[j];
      resumeGam[resumeChainIndex[j]].resize(gamSize);
      for(unsigned int k=0;k<gamSize;k++){
        f_resume >> resumeGam[resumeChainIndex[j]][k];
      }
    }
    f_resume.close();

  }else{
    if(!fixInit){

      standardize_matrix_gsl(mat_X);

      Double_Matrices Gam_step_regr_pvals;
      Gam_step_regr_pvals.Alloc_double_matrix(pY,
					  pX);
      Double_Matrices Gam_step_regr_SE;
      Gam_step_regr_SE.Alloc_double_matrix(pY,
				       pX);

      Double_Matrices Gam_step_regr_beta;
      Gam_step_regr_beta.Alloc_double_matrix(pY,
					 pX);

      double tolerance= 6.6835e-14;

      gsl_vector *current_outcome =gsl_vector_calloc(nX);
      gsl_vector *vect_residuals=gsl_vector_calloc(nX);
      gsl_vector *vect_p_value=gsl_vector_calloc(pX);
      gsl_vector *vect_beta_full=gsl_vector_calloc(pX);
      gsl_vector *vect_SE_full=gsl_vector_calloc(pX);

      cout << "***************************************************" << endl
          << "*************  Stepwise regression   *************" << endl
          << "***************************************************" << endl << endl;

      for(unsigned int outcome=0;outcome<pY;outcome++){
        vector < unsigned int > list_columns_X_gam;
        vector < unsigned int > list_columns_X_gam_bar;
        vector < unsigned int > is_var_in;

        is_var_in.resize(pX);
        for(unsigned int jj=0;jj<nConfounders;jj++){
          // Set up so confounders are always in
          is_var_in[jj]=1;
        }

    
        gsl_matrix_get_col(current_outcome,
		       mat_Y,
		       outcome);
        int stop=0;
        int loop=0;
        int n_loop_max=100;

        while(stop==0){
          get_list_var_in_and_out(list_columns_X_gam,
			      list_columns_X_gam_bar,
			      is_var_in);
      
   
          gsl_matrix *mat_X_gam=get_X_reduced_and_constant(list_columns_X_gam,
							mat_X);
      
          gsl_matrix *mat_X_gam_bar=get_X_reduced(list_columns_X_gam_bar,
					       mat_X);
          //Calculating p-values for all variables
          get_full_pvalues_and_beta(vect_p_value,
				vect_beta_full,
				vect_SE_full,
				mat_X_gam,
				mat_X_gam_bar,
				current_outcome,
				vect_residuals,
				vect_RMSE,
				tolerance,
				list_columns_X_gam,
				list_columns_X_gam_bar,
				outcome); 
          //Updating the list of var in
          stop=update_is_var_in(is_var_in,
			    list_columns_X_gam,
			    list_columns_X_gam_bar,
			    vect_p_value,
			    Pvalue_enter,
			    Pvalue_remove,
			    loop,
			    n_loop_max,
			    nConfounders);
          gsl_matrix_free(mat_X_gam);
          gsl_matrix_free(mat_X_gam_bar);
 
          loop++;
          list_columns_X_gam.clear();
          list_columns_X_gam_bar.clear();
      
          if(stop==1){
            get_list_var_in_and_out(list_columns_X_gam,
				list_columns_X_gam_bar,
				is_var_in);
          }
      
        }//end of while
        //Filling the output file


 
        store_model_per_outcome(Gam_step_regr,
			    list_columns_X_gam,
			    vect_p_value,
			    vect_beta_full,
			    vect_SE_full,
			    Gam_step_regr_pvals,
			    Gam_step_regr_SE,
			    Gam_step_regr_beta,
			    outcome);
    

        list_columns_X_gam.clear();
        is_var_in.clear();
      }

      cout << "Result From Step-Wise Regression" << endl;
      display_matrix_var_dim(Gam_step_regr);

      cout << endl;


      cout << "***************************************************" << endl
          << "**********  End of Stepwise regression  **********" << endl
          << "***************************************************" << endl << endl;

      gsl_vector_free(vect_residuals);
      gsl_vector_free(current_outcome);
      gsl_vector_free(vect_p_value);
      gsl_vector_free(vect_beta_full);
      gsl_vector_free(vect_SE_full);

      Gam_step_regr_pvals.Free_double_matrix();
      Gam_step_regr_SE.Free_double_matrix();
      Gam_step_regr_beta.Free_double_matrix();

    }else{

      // Read the initial gamma from file
      ifstream inputFile;
      inputFile.open(filename_init);
      if(!inputFile.is_open()){
        cout << "Input file not found" << endl;
        exit(-1);
      }
      unsigned int initSize;
      inputFile >> initSize;
      initGam.resize(initSize);
      for(unsigned int i=0;i<initSize;i++){
        inputFile >> initGam[i];
      }
      inputFile.close();

      cout << "********************************************************" << endl
          << "***************** Initial indices entered **************" << endl;
      for(unsigned int j=0;j<initGam.size();j++){
        cout << initGam[j] << " ";
      }
      cout << endl;
      cout << "********************************************************" << endl
          << "********************************************************" << endl;


      // Need to get the RMSE for prior estimate of k

      standardize_matrix_gsl(mat_X);

      gsl_vector *current_outcome =gsl_vector_calloc(nX);

      vector < unsigned int > list_columns_X_gam;
      vector < unsigned int > is_var_in(pX);

      unsigned int k=0;
      for(unsigned int j=0;j<pX;j++){
        // Set up so confounders are always in
        if(j<nConfounders){
          is_var_in[j]=1;
        }else if(j==initGam[k]){
          k++;
          is_var_in[j]=1;
        }else{
          is_var_in[j]=0;
        }
      }
      get_list_var_in(list_columns_X_gam,is_var_in);

      gsl_matrix* mat_X_gam=get_X_reduced_and_constant(list_columns_X_gam,
                                                            mat_X);

      double tolerance= 6.6835e-14;

      for(unsigned int outcome=0;outcome<pY;outcome++){
        gsl_matrix_get_col(current_outcome,mat_Y,outcome);
        getEstimateRMSE(mat_X_gam,
                      current_outcome,
                      outcome,
                      vect_RMSE,
                      tolerance);
      }

      gsl_matrix_free(mat_X_gam);
      gsl_vector_free(current_outcome);
    }
  }
  gsl_matrix_free(mat_X);

  // Re-read in the data (we need to do this to remove the standardisation)
  f_X.open(filename_in_mat_X,ios::in);
  f_X >> nX;
  f_X >> pX;

  gsl_matrix *mat_X_work2=gsl_matrix_alloc(nX,pX);

  for(unsigned int i=0;i<nX;i++){
    for(unsigned int j=0;j<pX;j++){
      double tmp;
      f_X >> tmp;
      gsl_matrix_set(mat_X_work2,i,j,tmp);
    }
  }
  f_X.close();

  gsl_matrix *mat_Y_work2=mat_Y;

  //Centering X and Y
  center_matrix_gsl(mat_X_work2);
  center_matrix_gsl(mat_Y_work2);
  if(standardizeFlag){
    // Standardize X matrix if selected
    standardize_matrix_gsl(mat_X_work2);
  }


  if(!postProcessOnly){
    cout << "**********************************************************" << endl
       << "****************** MOVES parameters **********************" << endl
       << "g-adaptative M-H" << endl
       << "\t-g_n_batch: " << g_n_batch_from_read << endl
       << "\t-g_AdMH_optimal: " << g_AdMH_optimal_from_read << endl
       << "\t-g_AdMH_ls: " << g_AdMH_ls_from_read << endl
       << "Crossover Move" << endl
       << "\tk_max: " <<k_max_from_read << endl
       << "Gibbs Move" << endl
       << "\tGibbs_n_batch: " <<Gibbs_n_batch << endl
       << "**********************************************************" << endl
       << "**********************************************************" << endl << endl;
 
    cout << "**********************************************************" << endl
        << "****************** TEMP parameters **********************" << endl
        << "b_t " << b_t_input << endl
        << "a_t_den_inf_5k " << a_t_den_inf_5k << endl
        << "a_t_den_5_10k " << a_t_den_5_10k << endl
        << "a_t_den_sup_10k " << a_t_den_sup_10k << endl
        << "temp_n_batch " << temp_n_batch << endl
        << "temp_optimal " << temp_optimal_input << endl
        << " M= [" << M_input[0] << " - " << M_input[1] << "]" << endl
        << "**********************************************************" << endl
        << "**********************************************************" << endl << endl;
  }

  //////////////////////////////////
  //  Setting up prior parameters
  //////////////////////////////////
  
  Prior_param PR;
  PR.set_PR_param(E_p_gam_from_read,
		  Sd_p_gam_from_read,
		  delta_from_read,
		  pX,
		  pY,
		  nX,
		  lambda,
		  vect_RMSE,
		  P_mutation_from_read,
		  P_sel_from_read,
		  P_csvr_r_from_read,
		  P_DR_from_read);
  cout << "HERE " << PR.k << endl;
  if(!postProcessOnly){
    PR.display_prior_param();
  }

  //////////////////////////////////
  //  Setting up g_AdMH parameters
  //////////////////////////////////
  
  g_AdMH *My_g_AdMH=new g_AdMH;
  if(!postProcessOnly){
    (*My_g_AdMH).set_g_AdMH(gSampleFlag,
			  g_n_batch_from_read,
			  g_AdMH_optimal_from_read,
			  g_AdMH_ls_from_read,
			  pX,
			  burn_in,
			  g_M_min_input,
			  g_M_max_input);

    if(resumeRun || extendRun){
      (*My_g_AdMH).ls=resumeLS;
    }
    (*My_g_AdMH).display_g_AdMH();
  }
  
  //////////////////////////////////
  //  Setting up DR parameters
  //////////////////////////////////

  DR *My_DR=new DR();
  if(!postProcessOnly){
    (*My_DR).set_DR(nb_chains);
    if(resumeRun || extendRun){
      (*My_DR).nb_calls=resumeDRNCalls;
      (*My_DR).nb_calls_adj=resumeDRNCallsAdj;
      for(unsigned int j=0;j<nb_chains;j++){
        for(unsigned int k=0;k<nb_chains;k++){
          (*My_DR).mat_moves_accepted[j][k]=resumeDRAccepted[j][k];
          (*My_DR).mat_moves_proposed[j][k]=resumeDRProposed[j][k];
        }
      }
    }
    (*My_DR).display_DR();
  }
  
  //////////////////////////////////
  //  Setting up CM parameters
  //////////////////////////////////
  vector <unsigned int > list_CM_moves_enabled_from_read;
  CM *My_CM=new CM;
  if(!postProcessOnly){
    list_CM_moves_enabled_from_read.push_back(1);
    list_CM_moves_enabled_from_read.push_back(2);
  
  
    (*My_CM).set_CM(k_max_from_read,
		  list_CM_moves_enabled_from_read);
  
    (*My_CM).display_CM();
  
    list_CM_moves_enabled_from_read.clear();
  }
  //////////////////////////////////
  //  Defining Move Monitor Object
  //////////////////////////////////

  Move_monitor *My_Move_monitor=new Move_monitor(nb_chains,
						 (*My_CM).n_possible_CM_moves);

  //////////////////////////////////
  //   Initializing the chains
  //////////////////////////////////
 
  vector < vector <unsigned int> > vect_gam;
  if(!postProcessOnly){
    cout << "Initialising chains" << endl;

    vect_gam.resize(nb_chains);
    for(unsigned int chain=0;chain<nb_chains;chain++){
      vect_gam[chain].resize(pX);
    }
  
    if(resumeRun || extendRun){
      for(unsigned int chain=0;chain<nb_chains;chain++){
        unsigned int k=0;
        for(unsigned int j=0;j<pX;j++){
          if(resumeGam[chain].size()>0){
            if(j<nConfounders){
              vect_gam[chain][j]=1;
            }else if(j==resumeGam[chain][k]){
              k++;
              vect_gam[chain][j]=1;
            }else{
              vect_gam[chain][j]=0;
            }
          }else{
            if(j<nConfounders){
              vect_gam[chain][j]=1;
            }else{
              vect_gam[chain][j]=0;
            }
          }
        }
      }

    }else{
      if(!fixInit){
        get_vect_gam_init(vect_gam,
		    Gam_step_regr,
		    iso_T_Flag,
		    maxPGamma,
		    RandomNumberGenerator);
      }else{
        for(unsigned int chain=0;chain<nb_chains;chain++){
          unsigned int k=0;
          for(unsigned int j=0;j<pX;j++){
            if(j<nConfounders){
              vect_gam[chain][j]=1;
            }else if(j==initGam[k]){
              k++;
              vect_gam[chain][j]=1;
            }else{
              vect_gam[chain][j]=0;
            }
          }
        }
      }
    }


    Gam_step_regr.clear();
  }

  Temperatures *t_tun=new Temperatures();
  Double_Matrices mat_log_marg;
  Double_Matrices mat_log_cond_post;
  gsl_permutation *MyPerm=gsl_permutation_calloc(pX);
  unsigned int sweep=0;
  vector < unsigned int > chain_idx;
  gsl_matrix *description_exchange_move=description_exch_moves(nb_chains);

  if(!postProcessOnly){
    //////////////////////////////////
    // Intializing chain temparature
    //////////////////////////////////

    (*t_tun).set_Temp_param(nb_chains,
			  pX,
			  b_t_input,
			  a_t_den_inf_5k,
			  a_t_den_5_10k,
			  a_t_den_sup_10k,
			  temp_n_batch,
			  M_input,
			  burn_in,
			  temp_optimal_input,
			  iso_T_Flag);
    M_input.clear();
    if(resumeRun || extendRun){
      for(unsigned int j=0;j<nb_chains;j++){
        (*t_tun).t[j]=resumeT[j];
      }
      (*t_tun).b_t=resumeBT;
    }
    (*t_tun).display_Temp_param();

    ///////////////////////////////////
    //   Declaring output matrices
    ///////////////////////////////////

    if(extendRun){
      n_sweeps+=nExtSweeps;
    }
    
    
    mat_log_marg.Alloc_double_matrix(nb_chains,
				     n_sweeps+1);
    
    mat_log_cond_post.Alloc_double_matrix(nb_chains,
					  n_sweeps+1);
    
    
    if(resumeRun || extendRun){
      sweep=resumeSweep;
    }


    //Defining the chain ID:
    //chain_idx[i], says on which line chain number i is located
    intialize_chain_idx(chain_idx,
		      nb_chains);
    if(resumeRun || extendRun){
      for(unsigned int c=0;c<nb_chains;c++){
        chain_idx[c]=resumeChainIndex[c];
      }
    }

    //description_exchange_move: each move is presented in columns
    // -line 1: first chain
    // -line 2: second chain
    // -line 3: Pbty of the move

  }
  endTime = std::clock();
  tmpTime = startTime;
  setupTime = (endTime-tmpTime)/(double)(CLOCKS_PER_SEC);


 
  if(!postProcessOnly){
    cout << "***************************************" << endl
        << "             FIRST SWEEP" << endl
        <<  "***************************************" << endl << endl;
    for(unsigned int chain=0;chain<nb_chains;chain++){
  
      //Initial Calculation of the logMarg and log_cond_post;
      vector < unsigned int > list_columns_X_gam;
      vector < unsigned int > list_columns_X_gam_bar;
    
      //Step 1 Getting X_gamma
      get_list_var_in_and_out(list_columns_X_gam,
			    list_columns_X_gam_bar,
			    vect_gam[chain_idx[chain]]);
    
      cout << "List_columns_X" << endl;
      for(unsigned int col=0;col<list_columns_X_gam.size();col++){
        cout << list_columns_X_gam[col] << " ";
      }
      cout << endl;
      unsigned int n_vars_in=list_columns_X_gam.size();
      cout << "n_vars_in " << n_vars_in << endl;

      gsl_matrix *matXGam = NULL;
      double logMargLik,logPost;
      if(n_vars_in>0){
        matXGam=get_X_reduced(list_columns_X_gam,
					   mat_X_work2);
      
        //Step 2: Calculate log-marginal
      }
      computeLogPosterior(logMargLik,
                          logPost,
                          matXGam,
                          mat_Y_work2,
                          PR,
                          gPriorFlag,
                          indepPriorFlag,
                          gSampleFlag,
                          lambda,
                          g,
                          pX,
                          n_vars_in,
                          nX,
                          pY,
                          cudaFlag);
      mat_log_marg.matrix[chain_idx[chain]][sweep]=logMargLik;
      mat_log_cond_post.matrix[chain_idx[chain]][sweep]=logPost;

      if(n_vars_in>0){
        gsl_matrix_free(matXGam);
      }

      cout << endl << "**************Results, chain " << chain << " -- sweep " << sweep << "***************" << endl;
      cout << "\tlog_marg " << mat_log_marg.matrix[chain_idx[chain]][sweep] << endl
          << "\tlog_cond_post " << mat_log_cond_post.matrix[chain_idx[chain]][sweep] << endl;
      cout << "********************************************************" << endl;
    
    
    
      list_columns_X_gam.clear();
      list_columns_X_gam_bar.clear();
    
    }//end of for chain.
  
    cout << "***************************************" << endl
        << "          END OF FIRST SWEEP" << endl
        <<  "***************************************" << endl << endl;
    if(!resumeRun  && !extendRun){
      print_main_results_per_sweep(f_out,
			      vect_gam,
			      chain_idx,
			      mat_log_marg,
			      mat_log_cond_post,
			      0);
    }
  }
  double cum_g=0.0;
  unsigned int count_g=0.0;
  if(resumeRun||postProcessOnly||extendRun){
    cum_g=resumeCumG;
    count_g=resumeCountG;
  }

  ////////////////////////////////////
  ////////////////////////////////////
  //   START OF THE ITERATIVE
  //   ALGORITHM
  ////////////////////////////////////
  ////////////////////////////////////

  vector <unsigned int> n_Models_visited;
  vector < vector <unsigned int> > List_Models;
  vector < vector <unsigned int> > Unique_List_Models;



  if(postProcessOnly){
    n_Models_visited.assign(resumeSweep+1,0);
  }else{
    n_Models_visited.assign(n_sweeps+1,0);
  }
  if(resumeRun||postProcessOnly||extendRun){
    if(resumeSweep>burn_in){
      List_Models.resize(resumeSweep-burn_in);
    }
    for(unsigned int i=0;i<resumeSweep+1;i++){
      n_Models_visited[i]=pastNModelsVisited[i];
      if(i>burn_in){
        unsigned int tmp = pastModels[i][0];
        List_Models[i-1-burn_in].resize(tmp+1);
        List_Models[i-1-burn_in][0]=tmp;
        for(unsigned int j=0;j<tmp;j++){
          List_Models[i-1-burn_in][j+1]=pastModels[i][j+1];
        }
      }
    }

  }

  if(!postProcessOnly){
    /////////////////////////////////////
    // Outputing features of the run
    /////////////////////////////////////
    
    f_out_features << "n\t" << nX << endl;
    f_out_features << "p\t" << pX << endl;
    f_out_features << "q\t" << pY << endl;
    f_out_features << "n_conf\t" << nConfounders << endl;
    f_out_features << "nsweeps\t" << n_sweeps << endl;
    f_out_features << "burn.in\t" << burn_in << endl;
    f_out_features << "Egam\t" << E_p_gam_from_read << endl;
    f_out_features << "Sgam\t" << Sd_p_gam_from_read << endl;
    f_out_features << "delta\t" << delta_from_read << endl;
    f_out_features << "top\t" << n_top_models << endl;
    f_out_features << "nb.chain\t" << nb_chains << endl;
    f_out_features << "cuda\t" << cudaFlag << endl;
    f_out_features << "history\t" << HistoryFlag << endl;
    f_out_features << "time\t" << Time_monitorFlag << endl;
    f_out_features << "med_RMSE\t" << sqrt(PR.k) << endl;
    f_out_features << "seed\t" <<MY_SEED << endl;

 
    for(sweep=resumeSweep+1;sweep<n_sweeps+1;sweep++){
      clock_t Time_start,Time_end;
      Time_start=clock();
 
      // Reset the permutation object
      gsl_permutation_init(MyPerm);

      n_Models_visited[sweep]=n_Models_visited[sweep-1];
      if(Log_Flag){
        cout << "***************************************" << endl
            << "             SWEEP #" << sweep << "/" << n_sweeps << endl
            << "***************************************" << endl << endl;
      }
      ////////////////////////////////////////////////////
      /////////////////// Local Moves  ///////////////////
      ////////////////////////////////////////////////////
      for(unsigned int chain=0;chain<nb_chains;chain++){
        mat_log_marg.matrix[chain][sweep]=mat_log_marg.matrix[chain][sweep-1];
        mat_log_cond_post.matrix[chain][sweep]=mat_log_cond_post.matrix[chain][sweep-1];
      }

      //Gibbs Moves
      if(sweep%Gibbs_n_batch==0){
        if(Log_Flag){
          cout << "Gibbs" << endl;
        }
        Gibbs_move(mat_log_marg,
                  mat_log_cond_post,
                  mat_X_work2,
                  mat_Y_work2,
                  t_tun,
                  MyPerm,
                  vect_gam,
                  sweep,
                  gPriorFlag,
                  indepPriorFlag,
                  gSampleFlag,
                  lambda,
                  g,
                  PR,
                  My_Move_monitor,
                  chain_idx,
                  n_Models_visited,
                  cudaFlag,
                  nConfounders,
                  maxPGamma,
                  RandomNumberGenerator);


      }

      ////Fast Scan Metropolis Hastings (FSMH)
    
      double local_move_rand=myrand(RandomNumberGenerator);
      if(nb_chains==1){
        local_move_rand=0.0;
      }
      if(local_move_rand<PR.Prob_mut){
        if(Log_Flag){
          cout << "FSMH" << endl;
        }
        FSMH_move(mat_log_marg,
		mat_log_cond_post,
		mat_X_work2,
		mat_Y_work2,
		t_tun,
		MyPerm,
		vect_gam,
		sweep,
		gPriorFlag,
		indepPriorFlag,
		gSampleFlag,
		lambda,
		g,
		PR,
		My_Move_monitor,
		chain_idx,
		n_Models_visited,
		cudaFlag,
		nConfounders,
		maxPGamma,
		RandomNumberGenerator);
 
      }
      else{
        ///////////////////////////////////////////////////////
        /////////////////////// CM Moves  /////////////////////
        ///////////////////////////////////////////////////////

        //Defining the number of CO move to simulate
        if(Log_Flag){
          cout << "CM" << endl;
        }
        Crossover_move(mat_log_marg,
		     mat_log_cond_post,
		     mat_X_work2,
		     mat_Y_work2,
		     t_tun,
		     vect_gam,
		     gPriorFlag,
		     indepPriorFlag,
		     gSampleFlag,
		     lambda,
		     g,
		     PR,
		     My_Move_monitor,
		     chain_idx,
		     sweep,
		     My_CM,
		     n_Models_visited,
		     cudaFlag,
		     maxPGamma,
		     RandomNumberGenerator);
      }

      ///////////////////////////////////////////////////////
      ///////////////////// Global Moves  ///////////////////
      ///////////////////////////////////////////////////////
      if(nb_chains>1){
        double global_move_rand=myrand(RandomNumberGenerator);
        if(sweep<=burn_in){
          global_move_rand=0.0;//during burn-in, DR is the only global move
        }
      
        if(global_move_rand<PR.Prob_DR){
	
          if(Log_Flag){
            cout << "DR" << endl;
          }
	
          DR_move(My_DR,
		chain_idx,
		mat_log_cond_post,
		t_tun,
		sweep,
		My_Move_monitor,
		RandomNumberGenerator);
	
        }else{
          if(Log_Flag){
            cout << "AE" << endl;
          }
          All_exchange_move(description_exchange_move,
			  chain_idx,
			  mat_log_cond_post,
			  t_tun,
			  sweep,
			  My_Move_monitor,
			  RandomNumberGenerator);
        }

      }
    
      /////////////////////////////////////////////////////
      ///////////////////// Sampling g  ///////////////////
      /////////////////////////////////////////////////////

      if(gSampleFlag){
        sample_g(mat_log_marg,
	       mat_log_cond_post,
	       My_g_AdMH,
	       t_tun,
	       vect_gam,
	       mat_X_work2,
	       mat_Y_work2,
	       gPriorFlag,
	       indepPriorFlag,
	       gSampleFlag,
	       lambda,
	       g,
	       PR,
	       sweep,
	       My_Move_monitor,
	       chain_idx,
	       n_Models_visited,
	       cudaFlag,
	       RandomNumberGenerator);


        if(Log_Flag){
          cout << "g = " << g << endl;
        }

      }
    
    
      ///////////////////////////////////////////////////
      ///////////////// Temp placement //////////////////
      ///////////////////////////////////////////////////
      if(nb_chains>1){
        if(sweep<=burn_in){
          if(((*My_DR).nb_calls_adj==(*t_tun).nbatch) || ((*My_DR).nb_calls==5*(*t_tun).nbatch)){
            unsigned int n_vars_in_last_chain=sum_line_std_mat(vect_gam,
							     chain_idx[nb_chains-1]);
	  

            //cout << "t_placement" << endl;
	  
            temp_placement(t_tun,
			 My_DR,
			 My_Move_monitor,
			 sweep,
			 n_vars_in_last_chain,
			 nX,
			 iso_T_Flag);
            //cout << "end -- t_placement" << endl;
	  
          }
        }
      }
 
      if(Log_Flag){
      
        display_summary_result_per_sweep(vect_gam,
				      chain_idx,
				      mat_log_marg,
				      mat_log_cond_post,
				      sweep,
				      t_tun,
				      nConfounders);
      }
    
    
      if(timeLimit>0){
        print_and_save_main_results_per_sweep(f_out,
					 f_out_n_vars_in,
					 f_out_n_models_visited,
					 f_out_log_cond_post_per_chain,
					 vect_gam,
					 List_Models,
					 chain_idx,
					 mat_log_marg,
					 mat_log_cond_post,
					 sweep,
					 burn_in,
					 n_Models_visited[sweep],
					 HistoryFlag);

        if(HistoryFlag){
          (*My_Move_monitor).print_move_monitor_per_sweep(f_out_FSMH,
                                                   f_out_CM,
                                                   f_out_AE,
                                                   f_out_DR,
                                                   f_out_g,
                                                   f_out_g_adapt,
                                                   f_out_Gibbs,
                                                   f_out_t_tun,
                                                   gSampleFlag,
                                                   iso_T_Flag,
                                                   sweep,
                                                   nb_chains);

        }
      }else{
        print_and_save_main_results_per_sweep(strStrOut,
                                                 strStrOutNVarsIn,
                                                 strStrOutNModelsVisited,
                                                 strStrOutLogCondPostPerChain,
                                                 vect_gam,
                                                 List_Models,
                                                 chain_idx,
                                                 mat_log_marg,
                                                 mat_log_cond_post,
                                                 sweep,
                                                 burn_in,
                                                 n_Models_visited[sweep],
                                                 HistoryFlag);

        if(HistoryFlag){
          (*My_Move_monitor).print_move_monitor_per_sweep(strStrOutFSMH,
                                                           strStrOutCM,
                                                           strStrOutAE,
                                                           strStrOutDR,
                                                           strStrOutG,
                                                           strStrOutGAdapt,
                                                           strStrOutGibbs,
                                                           strStrOutTTun,
                                                           gSampleFlag,
                                                           iso_T_Flag,
                                                           sweep,
                                                           nb_chains);

        }
      }

      if(sweep>burn_in){
        cum_g+=g;
        count_g++;
      }

      if(timeLimit>0){
        f_resume.open(resumeName.c_str(),ios::out);
        f_rng=fopen(rngFileName.c_str(),"wb");

        saveResumeFile(f_resume,f_rng,sweep,g,cum_g,count_g,vect_RMSE,t_tun,My_g_AdMH->ls,My_DR,
                          vect_gam,chain_idx,nConfounders,pY,RandomNumberGenerator);

        fclose(f_rng);
        f_resume.close();
      }

      Time_end=clock();
      double time_taken=((double)Time_end-Time_start)/CLOCKS_PER_SEC;
      if(Log_Flag){
        cout << "Sweep\tTime\t#models\tTime/model" << endl;
        cout << sweep << "\t"
            << time_taken << "\t"
            << n_Models_visited[sweep]-n_Models_visited[sweep-1] << "\t"
            << time_taken/(double)(n_Models_visited[sweep]-n_Models_visited[sweep-1])
            <<  endl;

        cout << "***************************************" << endl
            << "           END OF SWEEP #" << sweep << "/" << n_sweeps << endl
            <<  "***************************************" << endl << endl;
      }
    
      if(Time_monitorFlag){
        if(timeLimit>0){
          f_out_time << sweep << "\t" << time_taken << "\t" << time_taken/(double)(n_Models_visited[sweep]-n_Models_visited[sweep-1]) << endl;
        }else{
          strStrOutTime << sweep << "\t" << time_taken << "\t" << time_taken/(double)(n_Models_visited[sweep]-n_Models_visited[sweep-1]) << endl;
        }
      }

      currTime = time(NULL);
      double timeElapsed=((double)currTime-beginTime)/3600;
      //cout << "Time elapsed: " << timeElapsed << endl;
      if(timeLimit>0&&timeElapsed>timeLimit){
        // Time has passed the limit so we need to stop
        if(HistoryFlag){
          f_out.close();
          f_out_n_vars_in.close();
          f_out_log_cond_post_per_chain.close();
          f_out_n_models_visited.close();
          f_out_FSMH.close();
          f_out_CM.close();
          f_out_AE.close();
          f_out_DR.close();
          if(gSampleFlag){
            f_out_g.close();
            f_out_g_adapt.close();
          }
          f_out_Gibbs.close();
          if(iso_T_Flag==false){
            f_out_t_tun.close();
          }
        }
        if(Time_monitorFlag){
          f_out_time.close();
        }

        n_Models_visited.clear();

        for(unsigned int chain=0;chain<nb_chains;chain++){
          vect_gam[chain].clear();
        }
        vect_gam.clear();
        gsl_matrix_free(mat_X_work2);
        gsl_matrix_free(mat_Y_work2);

        gsl_vector_free(vect_RMSE);
        mat_log_marg.Free_double_matrix();
        mat_log_cond_post.Free_double_matrix();
        gsl_permutation_free(MyPerm);
        chain_idx.clear();
        gsl_matrix_free(description_exchange_move);
        delete (My_g_AdMH);
        delete (My_Move_monitor);
        delete (My_CM);
        delete (My_DR);
        delete (t_tun);

#if _CUDA_
        if(cudaFlag){
          //cublasShutdown();
          culaShutdown();
        }
#endif
	f_out_features << "last_sweep\t" << sweep << endl;
	f_out_features << "run_finished\t" << run_finished << endl;

        f_out_features.close();
        cout << "Run curtailed as time limit exceeded. Please run with -resume flag." << endl;
        return(0);

      }


    }//end of for sweep
  }

  // Now if we weren't writing as we went along, write the output
  if(!postProcessOnly){
    if(timeLimit<0){
      if(HistoryFlag){
        f_out << strStrOut.str();
        f_out_n_vars_in << strStrOutNVarsIn.str();
        f_out_log_cond_post_per_chain << strStrOutLogCondPostPerChain.str();
        f_out_n_models_visited << strStrOutNModelsVisited.str();
        f_out_Gibbs << strStrOutGibbs.str();
        f_out_FSMH << strStrOutFSMH.str();
        f_out_CM << strStrOutCM.str();
        f_out_AE << strStrOutAE.str();
        f_out_DR << strStrOutDR.str();
        if(gSampleFlag){
          f_out_g << strStrOutG.str();
          f_out_g_adapt << strStrOutGAdapt.str();
        }
        if(!iso_T_Flag){
          f_out_t_tun << strStrOutTTun.str();
        }
        if(Time_monitorFlag){
          f_out_time << strStrOutTime.str();
        }
      }
    }

    if(HistoryFlag){
      f_out.close();
      f_out_n_vars_in.close();
      f_out_log_cond_post_per_chain.close();
      f_out_n_models_visited.close();
      f_out_FSMH.close();
      f_out_CM.close();
      f_out_AE.close();
      f_out_DR.close();
      if(gSampleFlag){
        f_out_g.close();
        f_out_g_adapt.close();
      }
      f_out_Gibbs.close();
      if(!iso_T_Flag){
        f_out_t_tun.close();
      }
    }
    if(Time_monitorFlag){
      f_out_time.close();
    }

    cout << "***************************************" << endl
	 << "***************************************" << endl
	 << "           END OF SWEEPS" << endl
	 << "       # models evaluated: " << n_Models_visited[n_sweeps] << endl
	 << "***************************************" << endl
	 <<  "***************************************" << endl << endl;

    tmpTime = endTime;
    endTime = std::clock();
    mainLoopTime = (endTime-tmpTime)/(double)(CLOCKS_PER_SEC);
  }

  //////////////////
  //Post-Processing
  //////////////////

  //Step1: Calculating E(g);
  double mean_g=cum_g/(double)(count_g);
  //Step2: Get the Unique list of models and integrate log posterior
  
  vector< vector<double> >vect_marg_log_post;
  int pos_null_model;
  pos_null_model=getUniqueList(Unique_List_Models,
                                           List_Models,
                                           n_Models_visited,
                                           burn_in,
                                           pX,
                                           nConfounders);

  getLogPost(vect_marg_log_post,
            Unique_List_Models,
            mat_X_work2,
            mat_Y_work2,
            PR,
            gPriorFlag,
            indepPriorFlag,
            gSampleFlag,
            lambda,
            mean_g,
            cudaFlag);


  //Step4: Get the posterior of the model
  unsigned int n_unique = Unique_List_Models.size();
  gsl_permutation *idx_post_gam_sort=gsl_permutation_calloc(vect_marg_log_post.size());
  gsl_vector *vect_post_gam = gsl_vector_calloc(vect_marg_log_post.size());

  unsigned int n_retained_models=min(n_unique,n_top_models);


  getAndSortPostGam(vect_post_gam,
                   idx_post_gam_sort,
                   vect_marg_log_post);

  double nullMargLik=combineAndPrintBestModel(f_out_best_models,
					      idx_post_gam_sort,
					      vect_post_gam,
					      vect_marg_log_post,
					      Unique_List_Models,
					      n_retained_models,
					      pos_null_model,
					      Out_full_Flag,
					      nConfounders);
  
  
  getAndPrintMargGam(f_out_marg_gam,
		     Unique_List_Models,
		     vect_post_gam,
		     pX,
		     nConfounders);
  
  run_finished=true;

  f_out_features << "last_sweep\t" << sweep-1 << endl;
  f_out_features << "run_finished\t" << run_finished << endl;
  f_out_features << "null_marginal_likelihood\t" << nullMargLik << endl;

  tmpTime = endTime;
  endTime = std::clock();
  postProcessTime = (endTime-tmpTime)/(double)(CLOCKS_PER_SEC);
  cout << "Setup Time: " << setupTime  << endl;
  if(!postProcessOnly){
    cout << "Main Loop Time: " << mainLoopTime  << endl;
  }
  cout << "Post Processing Time: " << postProcessTime  << endl;

  cout << "***************************************" << endl
       << "Marginal likelihood null model:" << nullMargLik << endl
       << "***************************************" << endl;

  f_out_marg_gam.close();
  f_out_best_models.close();
  f_out_features.close();

  gsl_permutation_free(idx_post_gam_sort);
  gsl_vector_free(vect_post_gam);
    
  for(unsigned int line=0;line<List_Models.size();line++){
    List_Models[line].clear();
  }
  List_Models.clear();
  
  for(unsigned int line=0;line<Unique_List_Models.size();line++){
    Unique_List_Models[line].clear();
    vect_marg_log_post[line].clear();
  }
  
  Unique_List_Models.clear();


  n_Models_visited.clear();

  if(!postProcessOnly){
    for(unsigned int chain=0;chain<nb_chains;chain++){
      vect_gam[chain].clear();
    }
  }
  vect_gam.clear();  

  gsl_matrix_free(mat_X_work2);
  gsl_matrix_free(mat_Y_work2);
  
  gsl_vector_free(vect_RMSE);
  if(!postProcessOnly){
    mat_log_marg.Free_double_matrix();
    mat_log_cond_post.Free_double_matrix();
  }
  gsl_permutation_free(MyPerm);
  chain_idx.clear();
  gsl_matrix_free(description_exchange_move);
  delete (My_g_AdMH);
  delete (My_Move_monitor);
  delete (My_CM);
  delete (My_DR);
  delete (t_tun);

#if _CUDA_
  if(cudaFlag){
    //cublasShutdown();
    culaShutdown();
  }
#endif

  return 0; 
}

