/*
 *       $Id$
 *
 *       This source code is part of
 *
 *        G   R   O   M   A   C   S
 *
 * GROningen MAchine for Chemical Simulations
 *
 *            VERSION 2.0
 * 
 * Copyright (c) 1991-1997
 * BIOSON Research Institute, Dept. of Biophysical Chemistry
 * University of Groningen, The Netherlands
 * 
 * Please refer to:
 * GROMACS: A message-passing parallel molecular dynamics implementation
 * H.J.C. Berendsen, D. van der Spoel and R. van Drunen
 * Comp. Phys. Comm. 91, 43-56 (1995)
 *
 * Also check out our WWW page:
 * http://rugmd0.chem.rug.nl/~gmx
 * or e-mail to:
 * gromacs@chem.rug.nl
 *
 * And Hey:
 * Gnomes, ROck Monsters And Chili Sauce
 */
static char *SRCID_expfit_c = "$Id$";

#include <sysstuff.h>
#include <string.h>
#include <math.h>
#include "typedefs.h"
#include "smalloc.h"
#include "xvgr.h"
#include "futil.h"
#include "gstat.h"
#include "vec.h"
#include "statutil.h"
#include "rdgroup.h"

extern void mrqmin(real x[],real y[],real sig[],int ndata,real a[],
		   int ma,int lista[],int mfit,real **covar,real **alpha,
		   real *chisq,
		   void (*funcs)(real x,real a[],real *y,real dyda[],int na),
		   real *alamda);

extern void mrqmin_new(real x[],real y[],real sig[],int ndata,real a[], 
		       int ia[],int ma,real **covar,real **alpha,real *chisq, 
		       void (*funcs)(real, real [], real *, real [], int), 
		       real *alamda);
		       
static real myexp(real x,real A,real tau)
{
  if ((A == 0) || (tau == 0))
    return 0;
  return A*exp(-x/tau);
}
		   
static void exp_one_parm(real x,real a[],real *y,real dyda[],int na)
{
  /* Fit to function 
   *
   * y = exp(-a1 x)
   *
   */
   
  real e1;
  
  e1      = exp(-x/a[1]);
  *y      = e1;
  dyda[1] = x*e1/(a[1]*a[1]);
}

static void exp_two_parm(real x,real a[],real *y,real dyda[],int na)
{
  /* Fit to function 
   *
   * y = a2 exp(-x/a1)
   *
   */
   
  real e1;
  
  e1      = exp(-x/a[1]);
  *y      = a[2]*e1;
  dyda[1] = x*a[2]*e1/(a[1]*a[1]);
  dyda[2] = e1;
}

static void exp_3_parm(real x,real a[],real *y,real dyda[],int na)
{
  /* Fit to function 
   *
   * y = a2 exp(-x/a1) + (1-a2) exp(-x/a3)
   *
   */
   
  real e1,e2;
  
  e1      = exp(-x/a[1]);
  e2      = exp(-x/a[3]);
  *y      = a[2]*e1 + (1-a[2])*e2;
  dyda[1] = x*a[2]*e1/(a[1]*a[1]);
  dyda[2] = e1-e2;
  dyda[3] = x*(1-a[2])*e2/(a[3]*a[3]);
  /* fprintf(stderr,"exp3: x=%10.3e *y=%10.3e dyda=%10.3e %10.3e %10.3e\n",
    x,*y,dyda[1],dyda[2],dyda[3]);  */
}

/* lmfit_exp supports up to 3 parameter fitting of exponential functions */
static void lmfit_exp(int nframes,real x[],real y[],real dy[],real ftol,
		      real parm[],real dparm[],bool bVerbose,int nfitparm,
		      char *fix)
{
  typedef void (*myfitfn)(real x,real a[],real *y,real dyda[],int na);
  myfitfn fitfn[3] = { exp_one_parm, exp_two_parm, exp_3_parm };
  real chisq,ochisq,alamda;
  real *a,**covar,**alpha;
  bool bCont;
  int  i,j,ma,mfit,*lista,*ia;

  if ((nfitparm < 1) || (nfitparm > 3))
    fatal_error(0,"nfitparm = %d, should be in 1..3 (%s,%d)",
		nfitparm,__FILE__,__LINE__);

  ma=mfit=nfitparm;         /* number of parameters to fit */
  snew(a,ma+1);
  snew(covar,ma+1);
  snew(alpha,ma+1);
  snew(lista,ma+1);
  snew(ia,ma+1);
  for(i=1; (i<ma+1); i++) {
    lista[i] = i;
    ia[i] = i;
    snew(covar[i],ma+1);
    snew(alpha[i],ma+1);
  }
  if (fix != NULL) {
    fprintf(stderr,"Will keep %s fixed during fit procedure\n",fix);
    if (strcmp(fix,"tau1") == 0) 
      ia[1]=0;
    else if (strcmp(fix,"A") == 0) 
      ia[2]=0;
    else if (strcmp(fix,"tau2") == 0) 
      ia[3]=0;
  }
  fprintf(stderr,"%d parameter fit\n",mfit);

  /* Initial params */
  alamda = -1;    /* Starting value   */
  chisq  = 1e12;
  a[1]   = parm[0];    /* tau1 */
  if (nfitparm > 1)
    a[2]   = parm[1];  /* AA   */
  if (nfitparm > 2) 
    a[3]   = parm[2];  /* tau2 */

  j = 0;      
  if (bVerbose)
    fprintf(stderr,"%4s  %10s  %10s  %10s  %10s  %10s\n",
	    "Step","chi^2","Lambda","A1","A2","A3");
  do {
    ochisq = chisq;
    /* mrqmin(x-1,y-1,dy-1,nframes,a,ma,lista,mfit,covar,alpha,
     *   &chisq,fitfn[mfit-1],&alamda);
     */
    mrqmin_new(x-1,y-1,dy-1,nframes,a,ia,ma,covar,alpha,&chisq,
	       fitfn[mfit-1],&alamda);
     
    if (bVerbose) {
      fprintf(stderr,"%4d  %10.5e  %10.5e  %10.5e",
	      j,chisq,alamda,a[1]);
      if (mfit > 1)
	fprintf(stderr,"  %10.5e",a[2]);
      if (mfit > 2)
	fprintf(stderr,"  %10.5e",a[3]);
      fprintf(stderr,"\n");
    }
    j++;
    bCont = ((fabs(ochisq - chisq) > fabs(ftol*chisq)) ||
	     ((ochisq == chisq)));
  } while (bCont && (alamda != 0.0) && (j < 50));
  if (bVerbose)
    fprintf(stderr,"\n");
    
  /* Now get the covariance matrix out */
  alamda = 0;

  /*  mrqmin(x-1,y-1,dy-1,nframes,a,ma,lista,mfit,covar,alpha,
   * &chisq,fitfn[mfit-1],&alamda);
   */
  mrqmin_new(x-1,y-1,dy-1,nframes,a,ia,ma,covar,alpha,&chisq,
	     fitfn[mfit-1],&alamda);

  for(j=0; (j<mfit); j++) {
    parm[j]  = a[j+1];
    dparm[j] = covar[j+1][j+1];
  }
	     
  for(i=0; (i<ma+1); i++) {
    sfree(covar[i]);
    sfree(alpha[i]);
  }
  sfree(a);
  sfree(covar);
  sfree(alpha);
  sfree(lista);
}

real do_lmfit(int ndata,real c1[],real sig[],real dt,
	      real begintimefit,real endtimefit,
	      char *fitfn,char *fittitle,bool bVerbose,int nfitparm,
	      real fit[],real fitparms[],char *fix)
{
  FILE *fp;
  char buf[32];

  int  i,j,nfitpnts;
  real integral,fittedfunc;
  real *parm,*dparm;
  real AA,tau1,tau2,srAA,srtau1,srtau2;  
  real *x,*y,*dy;
  real ftol = 1e-4;

  fprintf(stderr,"There are %d points to fit %d vars!\n",
	  ndata,nfitparm);
  fprintf(stderr,"Fit from %g thru %g, dt=%g\n",
	  begintimefit,endtimefit,dt);
  
  snew(x,ndata);
  snew(y,ndata);
  snew(dy,ndata);

  j=0;
  for(i=0; (i<ndata); i++) {
    if ( ((dt*i) >= begintimefit) && ((dt*i) <= endtimefit) ) {
      x[j]=dt*i;
      y[j]=c1[i];

      /* mrqmin does not like sig to be zero */
      if (sig[i]<1.0e-7)
	sig[i]=1.0e-7;
      dy[j]=sig[i];
#ifdef DEBUG 
      fprintf(stderr,"j= %d, i= %d, x= %g, y= %g, dy= %g\n",
	      j,i,x[j],y[j],dy[j]);
#endif 
      j++;
    }
  }
  nfitpnts=j;
  if (j < nfitparm) {
    fprintf(stderr,"Not enough data points for fitting!\n");
    integral = 0;
  }
  else {
    snew(parm,4);
    snew(dparm,4);

    parm[0]=parm[1]=parm[2] = 1.0;
    if (fitparms) {
      parm[0]=fitparms[0];
      if (nfitparm == 2)
	parm[1]=fitparms[1];
      else {
	parm[1]=fitparms[1];
	parm[2]=fitparms[2];
      }
    }
    
    lmfit_exp(nfitpnts,x,y,dy,ftol,parm,dparm,bVerbose,nfitparm,fix);
    
    tau1 = parm[0];
    srtau1 = dparm[0];
    if (nfitparm > 1) {
      AA = parm[1];
      srAA = dparm[1];
    }
    else 
      AA = 1.0;
    if (nfitparm > 2) {
      tau2 = parm[2];
      srtau2 = dparm[2];
    }
    else
      tau2 = 0.0;
    
    /* Compute the integral from begintimefit to endtimefit
     */
    integral=(tau1*(myexp(begintimefit,AA,  tau1)   - 
		    myexp(endtimefit,  AA,  tau1)) +
	      tau2*(myexp(begintimefit,1-AA,tau2) - 
		    myexp(endtimefit,  1-AA,tau2)));
    
    /* Generate THE output */
    if (bVerbose) {
      fprintf(stderr,"FIT: # points used in fit is: %d\n",nfitpnts);
      fprintf(stderr,"FIT: %21s%21s%21s\n",
	      "   A      ","tau1 (ps)    ","tau2 (ps)     ");
      fprintf(stderr,"FIT: ------------------------------------------------------------\n");
      fprintf(stderr,"FIT: %8.3g +/- %8.3g%9.4g +/- %8.3g%8.3g +/- %8.3g\n",
	      AA,srAA,tau1,srtau1,tau2,srtau2);
      fprintf(stderr,"FIT: Integral (calc with fitted function) from %g to %g ps is: %g\n",
	      begintimefit,endtimefit,integral);
      
      sprintf(buf,"test%d.xvg",nfitpnts);
      fp = xvgropen(buf,"C(t) + Fit to C(t)","Time (ps)","C(t)");
      fprintf(fp,"# AA = %g, tau1 = %g, tau2 = %g\n",AA,tau1,tau2);
      for(j=0; (j<ndata); j++) {
	fittedfunc = myexp(dt*j,AA,tau1) + myexp(dt*j,1-AA,tau2);
	fprintf(fp,"%10.5e  %10.5e  %10.5e\n",dt*j,c1[j],fittedfunc);
      }
      fclose(fp);
    }
    /* Now copy the fitted func into the fit array, if not null */
    if (fit)
      for(i=0; (i<ndata); i++) 
	fit[i] = myexp(i*dt,AA,tau1) + myexp(i*dt,1-AA,tau2);
  }
  
  fitparms[0]=tau1;
  fitparms[1]=AA;
  fitparms[2]=tau2; 

  sfree(x);
  sfree(y);
  sfree(dy);
  
  return integral;
}

void do_logfit(int ndata,real c1[],real dt,real begintimefit,real endtimefit,
	       char *fitfn,char *fittitle)
{
  FILE *fitfp;
  int i,n;
  real *time,*y,*acf;
  real aa,bb,A,tau;

  fprintf(stderr,"Will fit data from %g (ps) to %g (ps).\n\n",
	  begintimefit,endtimefit);

  snew(time,ndata);   /* allocate the maximal necessary space */
  snew(y,ndata);
  snew(acf,ndata);
  n=0;

  for(i=0; (i<ndata); i++) {
    if ( (dt*i >= begintimefit) && (dt*i <= endtimefit) ) {
    time[n]=dt*i;
    y[n]=log(c1[i]);
    acf[n]=c1[i];
    n++;
    }
  }
  lsq_y_ax_b(n,time,y,&aa,&bb);
  A=exp(bb);
  tau=-1.0/aa;
  fprintf(stderr,"# of data points used in the fit is : %d\n",n);
  fprintf(stderr,"Fitted to y=exp(ax+b):\n");
  fprintf(stderr,"a = %10.5f\t b = %10.5f\n",aa,bb);
  fprintf(stderr,"Fitted to y= A exp(-x/tau):\n");
  fprintf(stderr,"A  = %10.5f\t tau  = %10.5f\n", A, tau);

  fitfp=xvgropen(fitfn,fittitle,"Time (ps)","C(t)");
  fprintf(fitfp,"#\n#Fitted to y= A exp(-x/tau):\n");
  fprintf(fitfp,"#A  = %10.5f\t tau  = %10.5f\n#\n", A, tau);
  fprintf(fitfp,"#\n#time\t acf\t fit\n");
  for(i=0; (i<n); i++) {
    fprintf(fitfp,"%g\t %g\t %g\n",time[i],acf[i], A*exp(-(time[i]/tau)));
  }
  ffclose(fitfp);
  sfree(time);
  sfree(y);
  sfree(acf);
}

void do_expfit(int ndata,real c1[],real dt,real begintimefit,real endtimefit)
{
  int i,n;
  real *x,*y,*Dy;
  real aa,bb,saa,sbb,A,tau,dA,dtau;

  fprintf(stderr,"Will fit data from %g (ps) to %g (ps).\n",
	  begintimefit,endtimefit);

  snew(x,ndata);   /* allocate the maximum necessary space */
  snew(y,ndata);
  snew(Dy,ndata);
  n=0;

  for(i=0; (i<ndata); i++) {
    if ( (dt*i >= begintimefit) && (dt*i <= endtimefit) ) {
      x[n]=dt*i;
      y[n]=c1[i];
      Dy[n]=0.5;
      fprintf(stderr,"n= %d, i= %d, x= %g, y= %g\n",n,i,x[n],y[n]);
      n++;
    }
  }
  fprintf(stderr,"# of data points used in the fit is : %d\n\n",n);
  expfit(n,x,y,Dy,&aa,&saa,&bb,&sbb);

  A=exp(aa);
  dA=exp(aa)*saa;
  tau=-1.0/bb;
  dtau=sbb/sqr(bb);
  fprintf(stderr,"Fitted to y=exp(a+bx):\n");
  fprintf(stderr,"a = %10.5f\t b = %10.5f",aa,bb);
  fprintf(stderr,"\n");
  fprintf(stderr,"Fitted to y=Aexp(-x/tau):\n");
  fprintf(stderr,"A  = %10.5f\t tau  = %10.5f\n",A,tau);
  fprintf(stderr,"dA = %10.5f\t dtau = %10.5f\n",dA,dtau);
}


void expfit(int n, real *x, real *y, real *Dy, real *a, real *sa, 
	    real *b, real *sb)
{
  real *w,*ly,A,SA,B,SB;
  int  i;
  real sum,xbar,ybar,Sxx,Sxy,wr2,chi2,gamma,Db;
  
#define ZERO 0.0
#define ONE 1.0
#define ONEP5 1.5
#define TWO 2.0
  
#define sqr(x) ((x)*(x))

  /*allocate memory */
  snew(w,n);
  snew(ly,n);

  /* Calculate weights and values of ln(y). */
  for(i=0;(i<n); i++){
    w[i]=sqr(y[i]/Dy[i]);
    ly[i]=log(y[i]);
  }
  
  /* The weighted averages of x and y: xbar and ybar. */
  sum=ZERO;
  xbar=ZERO;
  ybar=ZERO;
  for(i=0;(i<n);i++){
    sum+=w[i];
    xbar+=w[i]*x[i];
    ybar+=w[i]*ly[i];
  }
  xbar/=sum;
  ybar/=sum;
  
  /* The centered product sums Sxx and Sxy, and hence A and B. */
  Sxx=ZERO;
  Sxy=ZERO;
  for(i=0;(i<n);i++){
    Sxx+=w[i]*sqr(x[i]-xbar);
    Sxy+=w[i]*(x[i]-xbar)*(ly[i]-ybar);
  }
  B=Sxy/Sxx;
  A=ybar-B*xbar;
  
  /* Chi-squared (chi2) and gamma. */
  chi2=ZERO;
  gamma=ZERO;
  for(i=0;(i<n);i++){
    wr2=w[i]*sqr(ly[i]-A-B*x[i]);
    chi2+=wr2;
    gamma+=wr2*(x[i]-xbar);
  }
  
  /* Refined values of A and B. Also SA and SB. */
  Db=-ONEP5*gamma/Sxx;
  B+=Db;
  A-=ONEP5*chi2/sum-xbar*Db;
  SB=sqrt((chi2/(n-2))/Sxx);
  SA=SB*sqrt(Sxx/sum+sqr(xbar));
  *a=A;
  *b=B;
  *sa=SA;
  *sb=SB;
}


