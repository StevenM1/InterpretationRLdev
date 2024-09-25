## Scenarios 5 and 6
rm(list=ls())
library(DEoptim)
library(parallel)
library(pracma)

## custom code
library(emcAdapt)  ## <- package for fast RL updating, using some C-code, attached
source('./fMRI.R') ## <- some code to generate fMRI design matrices and the like
source('./simulation_functions.R')
palette(c("black", 'lightslateblue', 'dark orange'))

# Function to simulate RL data and fMRI data, and then estimate RL parameters and GLM parameter
simulate_fit_recover1 <- function(true_beta=1, true_gamma=1, nReps, nTrials, pReward, estimated_pnames=c('alpha', 'beta'), assumed_beta=NULL, assumed_gamma=NULL, alpha=0.1) {
  ## simulate multiple experiments to reduce the influence of simulation noise
  PEsthisbeta <- matrix(NA, nrow=nTrials, ncol=nReps)
  estimated_phis <- estimated_parameter <- matrix(NA, nrow=1, ncol=nReps)
  for(ii in 1:nReps) {
    out <- simulate_RL_softmax(alpha, beta=true_beta, nTrials, pReward, rewardSensitivity=true_gamma, simulateChoice=TRUE)
    PEsthisbeta[,ii] = out$PE   # true prediction errors
    
    # assume this generates the fMRI signal
    X1 <- make_design_matrix(nTrials, out$PE, TR=1, duration=.001)
    signal1 <- make_fMRI_signal(X1, beta_PE=1)   ## beta_PE = the 'phi' parameter in the paper, assumed to be 1
    
    # estimate RL parameters
    constants <- c()
    if(!is.null(assumed_beta)) constants <- c(constants, 'beta'=assumed_beta)
    if(!is.null(assumed_gamma)) constants <- c(constants, 'gamma'=assumed_gamma)
    lower_ranges <- c('alpha'=0, 'beta'=0, 'gamma'=0)
    upper_ranges <- c('alpha'=1, 'beta'=10, 'gamma'=10)
    out1 <- DEoptim(ll_func, 
                    lower=lower_ranges[estimated_pnames], 
                    upper=upper_ranges[estimated_pnames], 
                    pnames=estimated_pnames, rewards=out$rewards, choices=out$choices, constants=constants)
    updated1 <- emcAdapt::adapt.c.emc(out$rewards, 
                                      arguments=list(startValues=c(0,0), learningRates=matrix(out1$optim$bestmem[[1]], nrow=nTrials, ncol=2)))
    
    PEs1_fit <- apply(updated1$predictionErrors, 1, max, na.rm=TRUE)
    # make design matrix, use it to fit phi
    X2 <- make_design_matrix(nTrials=nTrials, PE=PEs1_fit, TR=1, duration=.001)
    beta_hat = fit_GLM(X2, signal1)
    estimated_phis[1,ii] = beta_hat[1]
    if('beta' %in% estimated_pnames | 'gamma' %in% estimated_pnames) {
      estimated_parameter[1,ii] = out1$optim$bestmem[[2]] 
    } else {
      estimated_parameter[1,ii] = out1$optim$bestmem[[1]] 
    }
  }
  if(dim(PEsthisbeta)[2]==1) {
    variances <- var(PEsthisbeta)
  } else {
    variances <- mean(apply(PEsthisbeta,1,var))
  }
  
  return(list(estimated_phis=estimated_phis, variances=variances, estimated_parameter=estimated_parameter))
}


# Scenario 5; Correct specification: Simulate with varying beta (gamma=1) -------------
betas <- seq(0.5, 10, .1) 
out <- parallel::mclapply(betas, function(x) simulate_fit_recover1(true_beta=x, 
                                                                   nReps=1, 
                                                                   nTrials=1e3, 
                                                                   pReward=c(.8, .2),
                                                                   estimated_pnames=c('alpha', 'beta'),
                                                                   assumed_gamma=1, true_gamma=1), mc.cores = 8)

estimated_phis1 <- sapply(out, function(x) x$estimated_phis)
estimated_betas1 <- sapply(out, function(x) x$estimated_parameter)
variances1 <- sapply(out, function(x) x$variances)

## Figure 5
pdf(file='./figures/scenario5_revision.pdf', width=7, height=2.5)
par(mfrow=c(1,3))
par(mar=c(3,3.5,2,.5), las=1, mgp=c(2,.5,0), lwd=2)
plot(betas,variances1, bty='l', xlab=expression(paste('True ', beta)), ylab='Prediction error variance', main='A')
legend('topleft', legend=paste0('r = ', round(cor(betas,variances1),2)))
plot(betas,estimated_betas1, bty='l', xlab=expression(paste('True ', beta)), ylab=expression(paste('Estimated ', beta)), main='B')
plot(betas,estimated_phis1, bty='l', xlab=expression(paste('True ', beta)), ylab=expression(paste('Estimated ', phi)), main='C', ylim=c(0,1.1))
abline(h=1, lty=2, col=2, lwd=2)
legend('bottomright', expression(paste('True ', phi)), bty='n', lty=c(2), col=2)
dev.off()


## (not in hte paper) What happens if we misspecify beta? I.e., we could assume it is 1...
betas <- seq(0.5, 10, .1) 
out <- parallel::mclapply(betas, function(x) simulate_fit_recover1(true_beta=x, 
                                                                   nReps=1, 
                                                                   nTrials=1e3, 
                                                                   pReward=c(.8, .2),
                                                                   estimated_pnames=c('alpha'),
                                                                   assumed_beta=1,
                                                                   assumed_gamma=1, true_gamma=1), mc.cores = 8)
estimated_phis2 <- sapply(out, function(x) x$estimated_phis)
estimated_alpha2 <- sapply(out, function(x) x$estimated_parameter)
variances2 <- sapply(out, function(x) x$variances)

par(mfrow=c(1,3))
par(mar=c(3,3.5,2,.5), las=1, mgp=c(2,.5,0), lwd=2)
plot(betas,variances2, bty='l', xlab=expression(paste('True ', beta)), ylab='Prediction error variance', main='A')
legend('topleft', legend=paste0('r = ', round(cor(betas,variances1),2)))
plot(betas,estimated_alpha2, bty='l', xlab=expression(paste('True ', beta)), ylab=expression(paste('Estimated ', alpha)), main='B')
plot(betas,estimated_phis2, bty='l', xlab=expression(paste('True ', beta)), ylab=expression(paste('Estimated ', phi)), main='C', ylim=c(0,1.1))
abline(h=1, lty=2, col=2, lwd=2)
legend('bottomright', expression(paste('True ', phi)), bty='n', lty=c(2), col=2)

# Also a bad idea


# Scenario 6; Incorrect specification: Simulate with varying gamma (and assume beta=1) -------------
gammas <- seq(0.5, 10, .1) 
out <- parallel::mclapply(gammas, function(x) simulate_fit_recover1(true_gamma=x,
                                                                   true_beta=1, 
                                                                   nReps=1, 
                                                                   nTrials=1e3, 
                                                                   pReward=c(.8, .2),
                                                                   estimated_pnames=c('alpha', 'beta'),
                                                                   assumed_gamma=1), mc.cores = 8)

estimated_phis3 <- sapply(out, function(x) x$estimated_phis)
estimated_betas3 <- sapply(out, function(x) x$estimated_parameter)
variances3 <- sapply(out, function(x) x$variances)


# Figure 6
pdf(file='./figures/scenario6_revision.pdf', width=7, height=2.5)
par(mfrow=c(1,3))
par(mar=c(3,3.5,2,.5), las=1, mgp=c(2,.5,0), lwd=2)
plot(gammas,variances3, bty='l', xlab=expression(paste('True ', gamma)), ylab='Prediction error variance', main='A')
legend('topleft', legend=paste0('r = ', round(cor(gammas,variances3),2)))

plot(gammas,estimated_betas3, bty='l', xlab=expression(paste('True ', gamma)), ylab=expression(paste('Estimated ', beta)), main='B')
axis(side=1, at=1, labels='', tck=0.035, col=2,lwd=2)
arrows(x0=1.5, x1=5, y0=0.5, y1=1.5, xpd=TRUE, length=0.05,col=2,lwd=2, code=1)
text(x=5.2, y=1.6, labels=expression(paste('Assumed ', gamma)), xpd=TRUE, pos=4)  
abline(h=1, lty=2, col=2, lwd=2)
legend('topleft', expression(paste('True ', beta)), bty='n', lty=c(2), col=2)

plot(gammas,estimated_phis3, bty='l', xlab=expression(paste('True ', gamma)), ylab=expression(paste('Estimated ', phi)), main='C')
abline(h=1, lty=2, col=2, lwd=2)
legend(6, 2.5, expression(paste('True ', phi)), bty='n', lty=c(2), col=2)
dev.off()

