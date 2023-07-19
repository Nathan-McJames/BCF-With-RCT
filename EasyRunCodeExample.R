#Just a simple R script to show how to run the code on github.
#Load some libraries that are needed.
library(mvtnorm)
library(dbarts)
library(Rcpp)
library(bcf)

#Set up the example from the Kallus paper
n_rct<-300
n_obs<-1000

x_rct<-runif(n_rct, -1, 1)

z_rct<-rbinom(n_rct, 1, 0.5)

z_obs<-rbinom(n_obs, 1, 0.5)

u_rct <- rnorm(n_rct, 0, 1)

#matrix for storing the x_obs and u_obs columns
x_u_obs<-matrix(0, nrow=n_obs, ncol=2)

for(i in 1:n_obs)
{
  x_u_obs[i,]<-rmvnorm(1, c(0, 0), sigma=matrix(c(1, z_obs[i]-0.5,
                                                  z_obs[i]-0.5, 1), byrow=T, ncol=2))
}

#get the columns needed
x_obs<-x_u_obs[,1]
u_obs<-x_u_obs[,2]


y_obs<-1+z_obs+x_obs+2*z_obs*x_obs+0.5*x_obs^2+0.75*z_obs*x_obs^2+u_obs+0.5*rnorm(n_obs, 0, 1)
y_rct<-1+z_rct+x_rct+2*z_rct*x_rct+0.5*x_rct^2+0.75*z_rct*x_rct^2+u_rct+0.5*rnorm(n_rct, 0, 1)

true_icates_obs<-0.75*x_obs^2+2*x_obs+1
true_icates_rct<-0.75*x_rct^2+2*x_rct+1

x_obs<-as.matrix(x_obs, ncol=1)
x_rct<-as.matrix(x_rct, ncol=1)
x_combined<-rbind(x_obs, x_rct)

y_combined<-c(y_obs, y_rct)

#indicates if somebody is in the treatment or control group
z_treat<-c(z_obs, z_rct)
#indicates if observational or rct data
z_in_rct<-c(rep(0, n_obs), rep(1, n_rct))

true_icates<-c(true_icates_obs, true_icates_rct)

#Run this line of code to get access to the main bcf function
sourceCpp(file = "~/YOUR_DIRECTORY_HERE/fast_obs_rct.cpp")

#estimate the propensity score
#not actually a probability because it is not scaled at the moment
p_mod<-bart(cbind(x_combined, z_in_rct), z_treat)
propensity_score<-colMeans(p_mod$yhat.train)


#set the values you want to use
n_tree_mu<-50
n_tree_mu_rct<-20
n_tree_tau<-20
n_tree_tau_rct<-20
n_iter<-2000
n_burn<-1000

rct_mod<-fast_rct_bcf(cbind(x_combined, propensity_score), #x_control covariates
                      y_combined, #y values
                      
                      z_in_rct, #by using an indicator that is 1 for those in the rct data, 
                      #the added mu and tau parts of the model only apply to rct people#
                      #If you replace z_in_rct with z_in_obs where z_in_obs is 1 for those in
                      #the observational data then the added mu and tau parts of the model will
                      #apply to those individuals instead.
                      #I will change the code and add an extra indicatior here which allows us to have the added mu 
                      #part for obs and the added tau part for rct or vice-versa as well.
                      
                      z_treat, #if received treatment
                      x_combined, #x_moderate covariates
                      
                      0.95, #alpha_mu
                      2, #beta_mu
                      0.25, #alpha_mu added part
                      3, #beta_mu added part
                      0.25, #alpha tau
                      3, #beta tau
                      0.25, #alpha_tau added part
                      3, #beta_tau added part
                      
                      n_tree_mu, #prior precision for the terminal node parameters
                      n_tree_mu_rct, #Here I use the number of trees because
                      n_tree_tau, #as the number of trees increases you want each tree
                      n_tree_tau_rct, #to contribute a smaller amount
                      #Can be adjusted if you want to try more/less restrictive priors
                      
                      3, #nu from gamma prior
                      0.1, #lambda from gamma prior
                      
                      n_iter, #number of iterations
                      n_tree_mu, #number of trees
                      n_tree_mu_rct,
                      n_tree_tau,
                      n_tree_tau_rct,
                      
                      1#minimum number of observations in a terminal node
                      #if a proposed grow/prune/change/swap operation results
                      #in a terminal node with less than this number
                      #of observations the proposal tree is automatically rejected
                      )


#These posterior samples include the burn in iterations so we need to remove them manually
rct_mod_mu_preds<-rowMeans(rct_mod$predictions_mu[,-c(1:n_burn)])
rct_mod_mu_rct_preds<-rowMeans(rct_mod$predictions_mu_rct[,-c(1:n_burn)])
rct_mod_tau_preds<-rowMeans(rct_mod$predictions_tau[,-c(1:n_burn)])
rct_mod_tau_rct_preds<-rowMeans(rct_mod$predictions_tau_rct[,-c(1:n_burn)])

#plot to test convergence
plot(rct_mod$sigmas, ylab="Residual Standard Deviation")

#Because the added tau part applies to the rct data we need to add it to the other tau part
#to get unbiased estimates I think.

icate_estimates<-rct_mod_tau_preds+rct_mod_tau_rct_preds

#plot to test if the predictions look ok.
#Black points from observational
#Pink points from rct
plot(true_icates, icate_estimates, xlab="True ICATES", ylab="Predicted ICATES", col=1+z_in_rct)
abline(0, 1)



