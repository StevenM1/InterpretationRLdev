rm(list=ls())
library(DEoptim)
library(parallel)
library(pracma)


## custom code
library(emcAdapt)  ## <- package for fast RL updating, using some C-code, attached
source('./fMRI.R') ## <- some code to generate fMRI design matrices and the like
source('./simulation_functions.R')

# Example simulation of RL paradigm
alpha <- .1
beta <- 1
gamma <- 1   # see section 'Beta and Gamma' and further for what this does

# experiment set-up
nTrials <- 1e3
pReward <- c(.8, .2)  # reward probabilities for choice1, choice2; assuming a single stimulus

out <- simulate_RL_softmax(alpha, beta, nTrials, pReward, rewardSensitivity=gamma)
Q <- out$Q
PE <- out$PE
choices <- out$choices
rewards <- out$rewards
mean(choices==1)   # accuracy

# sanity check plot
par(mfro=c(2,1))
plotRL(Q, PE, choices)
# looks OK
set.seed(1)

# This simulates a paradigm without choices (e.g., conditioning). We can contrast two learning rates here
par(mfcol=c(2,2))
out <- simulate_RL_softmax(alpha=.5, beta, nTrials, pReward, rewardSensitivity=gamma, simulateChoice=FALSE)
Q <- out$Q
PE <- out$PE
choices <- out$choices
rewards <- out$rewards
mean(choices==1)   # accuracy
plotRL(Q, PE, choices)

set.seed(1)
out <- simulate_RL_softmax(alpha=.1, beta, nTrials, pReward, rewardSensitivity=gamma, simulateChoice=FALSE)
Q <- out$Q
PE <- out$PE
choices <- out$choices
rewards <- out$rewards
mean(choices==1)   # accuracy
plotRL(Q, PE, choices)

palette(c("black", 'lightslateblue', 'dark orange'))

# Some background checks --------------------------------------------------
# What about the inverse temperature parameter?
betas <- seq(0.001, 10,  0.1)#) .025)
nReps <- 1
variances <- vector(mode='numeric', length=length(betas))
estimated_phis <- matrix(NA, nrow=length(betas), ncol=nReps)

simulate_fit_recover1 <- function(beta, nReps, nTrials, pReward) {
  ## simulate multiple experiments to reduce the influence of simulation noise
  PEsthisbeta <- matrix(NA, nrow=nTrials, ncol=nReps)
  estimated_phis <- estimated_betas <- matrix(NA, nrow=1, ncol=nReps)
  for(ii in 1:nReps) {
    out <- simulate_RL_softmax(.1, beta, nTrials, pReward)
    PEsthisbeta[,ii] = out$PE   # true prediction errors
    # assume this generates the signal
    X1 <- make_design_matrix(nTrials, out$PE, TR=1, duration=.001)
    signal1 <- make_fMRI_signal(X1, beta_PE=1)   ## beta_PE = the 'phi' parameter in the paper, assumed to be 1
    
    # estimate RL parameters, assuming outcome sensitivity is fixed
    out1 <- DEoptim(ll_func, lower=c(0,0), upper=c(1,10), pnames=c('alpha', 'beta'), rewards=out$rewards, choices=out$choices, constants=c('gamma'=1))
    updated1 <- emcAdapt::adapt.c.emc(out$rewards, 
                                      arguments=list(startValues=c(0,0), learningRates=matrix(out1$optim$bestmem[[1]], nrow=nTrials, ncol=2)))
    
    PEs1_fit <- apply(updated1$predictionErrors, 1, max, na.rm=TRUE)
    # make design matrix, use it to fit phi
    X2 <- make_design_matrix(nTrials=nTrials, PE=PEs1_fit, TR=1, duration=.001)
    beta_hat = fit_GLM(X2, signal1)
    estimated_phis[1,ii] = beta_hat[1]
    estimated_betas[1,ii] = out1$optim$bestmem[[2]]
  }
  if(dim(PEsthisbeta)[2]==1) {
    variances <- var(PEsthisbeta)
  } else {
    variances <- mean(apply(PEsthisbeta,1,var))
  }
  
  return(list(estimated_phis=estimated_phis, variances=variances, estimated_betas=estimated_betas))
}

# simulate_fit_recover
betas <- seq(0.5, 10, .1) 
out <- parallel::mclapply(betas, function(x) simulate_fit_recover1(beta=x, nReps=1, nTrials=1e3, pReward=pReward),
                          mc.cores = 8)

estimated_phis1 <- sapply(out, function(x) x$estimated_phis)
estimated_betas1 <- sapply(out, function(x) x$estimated_betas)
variances1 <- sapply(out, function(x) x$variances)
# par(mfrow=c(1,3))
# plot(betas,variances)
# plot(betas,estimated_betas1)
# plot(betas,estimated_phis)


simulate_fit_recover2 <- function(gamma, nReps, nTrials, pReward) {
  ## simulate multiple experiments to reduce the influence of simulation noise
  PEsthisbeta <- matrix(NA, nrow=nTrials, ncol=nReps)
  estimated_phis <- matrix(NA, nrow=1, ncol=nReps)
  estimated_betas <- matrix(NA, nrow=1, ncol=nReps)
  for(ii in 1:nReps) {
    out <- simulate_RL_softmax(.1, beta=1, rewardSensitivity=gamma, nTrials, pReward)
    PEsthisbeta[,ii] = out$PE   # true prediction errors
    # assume this generates the signal
    X1 <- make_design_matrix(nTrials, out$PE, TR=1, duration=.001)
    signal1 <- make_fMRI_signal(X1, beta_PE=1)   ## beta_PE = the 'phi' parameter in the paper, assumed to be 1
    
    # estimate RL parameters, assuming outcome sensitivity is fixed
    out1 <- DEoptim(ll_func, lower=c(0,0), upper=c(1,10), pnames=c('alpha', 'beta'), rewards=out$rewards, choices=out$choices, constants=c('gamma'=1))
    updated1 <- emcAdapt::adapt.c.emc(out$rewards, 
                                      arguments=list(startValues=c(0,0), learningRates=matrix(out1$optim$bestmem[[1]], nrow=nTrials, ncol=2)))
    
    PEs1_fit <- apply(updated1$predictionErrors, 1, max, na.rm=TRUE)
    # make design matrix, use it to fit phi
    X2 <- make_design_matrix(nTrials=nTrials, PE=PEs1_fit, TR=1, duration=.001)
    beta_hat = fit_GLM(X2, signal1)
    estimated_phis[1,ii] = beta_hat[1]
    estimated_betas[1,ii] = out1$optim$bestmem[[2]]
  }
  if(dim(PEsthisbeta)[2]==1) {
    variances <- var(PEsthisbeta)
  } else {
    variances <- mean(apply(PEsthisbeta,1,var))
  }
  
  return(list(estimated_phis=estimated_phis, variances=variances, estimated_betas=estimated_betas))
}
# undebug(simulate_fit_recover2)
simulate_fit_recover2(5, 1, 1000, pReward)

gammas <- seq(0.5, 10, .1) 
out2 <- parallel::mclapply(gammas, function(x) simulate_fit_recover2(gamma=x, nReps=1, nTrials=1e3, pReward=pReward),
                          mc.cores = 8)

estimated_phis2 <- sapply(out2, function(x) x$estimated_phis)
estimated_betas2 <- sapply(out2, function(x) x$estimated_betas)
variances2 <- sapply(out2, function(x) x$variances)


for(figtype in c('pdf', 'png')) {
  if(figtype == 'pdf') pdf(file='./figures/scenario5.pdf', width=7, height=2.5)
  if(figtype == 'png') png(file='./figures/scenario5.png', width=7, height=2.5, units='in', res=175)
  par(mfrow=c(1,3))
  par(mar=c(3,3.5,2,.5), las=1, mgp=c(2,.5,0), lwd=2)
  plot(betas,variances1, bty='l', xlab=expression(paste('True ', beta)), ylab='Prediction error variance', main='A')
  legend('topleft', legend=paste0('r = ', round(cor(betas,variances1),2)))
  plot(betas,estimated_betas1, bty='l', xlab=expression(paste('True ', beta)), ylab=expression(paste('Estimated ', beta)), main='B')
  plot(betas,estimated_phis1, bty='l', xlab=expression(paste('True ', beta)), ylab=expression(paste('Estimated ', phi)), main='C', ylim=c(0,1.1))
  abline(h=1, lty=2, col=2, lwd=2)
  legend('bottomright', expression(paste('True ', phi)), bty='n', lty=c(2), col=2)
  dev.off()

  if(figtype == 'pdf') pdf(file='./figures/scenario6.pdf', width=7, height=2.5)
  if(figtype == 'png') png(file='./figures/scenario6.png', width=7, height=2.5, units='in', res=175)
  par(mfrow=c(1,3))
  par(mar=c(3,3.5,2,.5), las=1, mgp=c(2,.5,0), lwd=2)
  plot(gammas,variances2, bty='l', xlab=expression(paste('True ', gamma)), ylab='Prediction error variance', main='A')
  legend('topleft', legend=paste0('r = ', round(cor(gammas,variances2),2)))
  plot(gammas,estimated_betas2, bty='l', xlab=expression(paste('True ', gamma)), ylab=expression(paste('Estimated ', beta)), main='B')
  plot(gammas,estimated_phis2, bty='l', xlab=expression(paste('True ', gamma)), ylab=expression(paste('Estimated ', phi)), main='C')
  abline(h=1, lty=2, col=2, lwd=2)
  legend(6, 2.5, expression(paste('True ', phi)), bty='n', lty=c(2), col=2)
  dev.off()
}


