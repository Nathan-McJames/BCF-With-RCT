#include <Rcpp.h>
using namespace Rcpp;

/*
 * This is a Bayesian linear regression model
 * that uses a Gibbs sampler to estimate the following model:
 * Y_i = B0 + B1*X_i + Epsilon_i
 * 
 * Where Epsilon_i is normally distributed with mean 0 and precision tau=1/sigma^2
 * 
 * BUT......
 * 
 * We have two groups of people:
 * 
 * Group G1 that helps to inform the intercept B0 AND slope B1
 * Members of this group have a power posterior of eta = 1 when estimating all model parameters
 * 
 * Group G2 that informs the slope FULLY, and the intercept PARTIALLY 
 * Members of this group have a power posterior of 0<=eta<=1 when estimating the intercept
 * But they have a power posterior of 1 for all other parameters which they contribute to as usual
 */


double sample_b0(NumericVector y, //outcome y
                 NumericVector x, //predictor variable x
                 NumericVector eta_power, //power of the posterior for each observation
                 double b1, //current slope estimate from current iteration
                 double tau, //current precision estimate from current iteration 
                 double mu_0, //prior mean of intercept parameter B0
                 double tau_0) //prior precision of intercept parameter B0
{
  /*
   * This time, instead of completely removing observations from group 2,
   * we weight their contribution by an amount eta
   */
  int n_total = y.size(); //get total number of observations
  double n_weighted = sum(eta_power); //Number of observations weighted by eta
  double precision = tau_0 + tau * n_weighted; 
  double mean = tau_0 * mu_0;
  for(int i=0; i<n_total; i++)
  {
    mean +=  tau * eta_power[i] * (y[i] - b1 * x[i]); 
  }
  mean /= precision;
  return R::rnorm(mean, 1 / sqrt(precision));
}


double sample_b1(NumericVector y, //outcome y
                 NumericVector x, //predictor x
                 double b0, //current intercept estimate from current iteration
                 double tau, //current precision estimate from current iteration
                 double mu_1, //prior mean for slope parameter
                 double tau_1) //prior precision for slope parameter
{
  //Here both groups contribute to the estimation of the slope
  //we do not need to do any sub-setting
  int n = y.size();
  double precision=tau_1;
  for(int i=0; i<n; i++)
  {
    precision += tau * x[i]*x[i]; 
  }
  double mean = tau_1 * mu_1;
  for(int i=0; i<n; i++)
  {
    mean += tau * ((y[i] - b0) * x[i]); 
  }
  mean /= precision;
  return R::rnorm(mean, 1 / sqrt(precision));
}



double sample_tau(NumericVector y, 
                 NumericVector x,
                 double b0,
                 double b1,
                 double alpha,
                 double beta)
{
  //Here, all observations contribute to estimating the precision tau=1/sigma^2
  //So no sub-setting is required again
  double n = y.size();
  double alpha_new=alpha+n/2;
  NumericVector resid(n);
  for(int i=0; i<n; i++)
  {
    resid[i]=y[i] - b0 - b1 * x[i]; 
  }
  double beta_new = beta;
  for(int i=0; i<n; i++)
  {
    beta_new+=(resid[i] * resid[i]) / 2; 
  }
  return R::rgamma(alpha_new, 1/beta_new);
}


// [[Rcpp::export]]
NumericVector gibbs(NumericVector y, 
                    NumericVector x,
                    NumericVector eta_power,
                    int iters,
                    double b0_init,
                    double b1_init,
                    double tau_init,
                    double mu_0,
                    double tau_0,
                    double mu_1,
                    double tau_1,
                    double alpha,
                    double beta)
{
  //For storing the samples
  NumericMatrix trace_b0_b1_tau(iters, 3);
  
  //Set initial values of intercept and slope and precision
  double b0=b0_init;
  double b1=b1_init;
  double tau=tau_init;
  
  //Go through all iterations
  for(int iter=0; iter<iters; iter++)
  {
    //Update intercept B0 using ONLY group 1 data
    b0 = sample_b0(y, x, eta_power, b1, tau, mu_0, tau_0);
    //Update slope with all data
    b1 = sample_b1(y, x, b0, tau, mu_1, tau_1);
    //Update precision with all data
    tau = sample_tau(y, x, b0, b1, alpha, beta);
    
    trace_b0_b1_tau(iter, 0)=b0;
    trace_b0_b1_tau(iter, 1)=b1;
    trace_b0_b1_tau(iter, 2)=tau;
  }
    
  return trace_b0_b1_tau;
}




/*** R
set.seed(1234567)

n_group_1<-300
n_group_2<-300

true_intercept<-5
true_slope<-2

#Create Data From Group 1
x_group_1<-runif(n_group_1, 0, 4)
y_group_1<-true_intercept+true_slope*x_group_1+rnorm(n_group_1)

#Create Data From Group 2
x_group_2<-runif(n_group_2, 6, 10)
y_group_2<-true_intercept+true_slope*x_group_2+rnorm(n_group_2, 0, 5)

combined_y<-c(y_group_1, y_group_2)
combined_x<-c(x_group_1, x_group_2)
eta_power<-c(rep(1, n_group_1), rep(0, n_group_2))

samples<-gibbs(combined_y, 
               combined_x,  
               eta_power,
               10000, 
               0, 
               0, 
               0.1, 
               0, 
               0.1, 
               0, 
               0.1, 
               2, 
               1)


plot(combined_x, combined_y, col=1+eta_power, xlab="X Values", ylab="Y Values",
     main="Plot of Resulting Regression Line")
abline(mean(samples[,1]), mean(samples[,2]))

hist(samples[-c(1:5000),1], xlab="Intercept Samples", main="")
hist(samples[-c(1:5000),2], xlab="Slope Samples", main="")
hist(sqrt(1/samples[-c(1:5000),3]), xlab="SD Samples", main="")

mean(samples[-c(1:5000),1])
*/
