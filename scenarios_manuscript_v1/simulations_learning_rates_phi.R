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


# Quick parameter recovery for sanity check -------------------------------
nDatasets <- 50
parrec.df <- data.frame(dataset=1:nDatasets, alpha=runif(nDatasets), beta=runif(nDatasets, 0, 10), gamma=1)
simulateAndRecover <- function(pars, pnames=c('alpha', 'beta', 'gamma'), constants=c('gamma'=1)) {
  alpha <- pars[[1]]
  beta <- pars[[2]]
  gamma <- pars[[3]]
  out <- simulate_RL_softmax(alpha, beta, nTrials, pReward, rewardSensitivity=gamma)
  Q <- out$Q
  PE <- out$PE
  choices <- out$choices
  rewards <- out$rewards
  
  # what's the LL of the true parameters?
  ll.true <- ll_func(c(alpha, beta, gamma), pnames=c('alpha', 'beta', 'gamma'), rewards=rewards, choices=choices, constants=NULL)
  
  # fit -- we need to fix one parameter to a constant here
  pnames <- pnames[!grepl(names(constants), pnames)]
  out <- DEoptim(ll_func, 
                 lower=setNames(c(0,0), pnames), upper=setNames(c(1,10), pnames), 
                 rewards=rewards, choices=choices, pnames=pnames, constants=constants)
  
  return(c('ll.true'=ll.true, setNames(out$optim$bestmem, paste0(names(out$optim$bestmem), '.fit')), 'll.fit'=out$optim$bestval))
}

out <- mclapply(1:nrow(parrec.df), function(x) simulateAndRecover(parrec.df[x,c('alpha', 'beta', 'gamma')]), mc.cores = 8)
out <- data.frame(do.call(rbind, out))
parrec.df <- cbind(parrec.df, out)

par(mfrow=c(2,2))
plot(parrec.df$alpha, parrec.df$alpha.fit, xlab='Data-generating alpha', ylab='Estimated alpha')
plot(parrec.df$beta, parrec.df$beta.fit, xlab='Data-generating beta', ylab='Estimated beta')
plot(parrec.df$ll.true, parrec.df$ll.fit, xlab='LL of data-generating pars', ylab='LL of estimated pars')

# yey, all looks good

# Some background checks --------------------------------------------------
### 1. How does the variance of PEs depend on alpha?
alphas <- seq(0.001, 1, .025)
nReps <- 20
variances <- vector(mode='numeric', length=length(alphas))
for(i in 1:length(alphas)) {
	## simulate multiple experiments to reduce the influence of simulation noise
	PEsthisalpha <- matrix(NA, nrow=nTrials, ncol=nReps)
	for(ii in 1:nReps) {
		out <- simulate_RL_softmax(alphas[i], beta, nTrials, pReward)
		PEsthisalpha[,ii] = out$PE
	}
	variances[i] <- mean(apply(PEsthisalpha,1,var))
}
par(mfrow=c(2,2))
plot(alphas, variances)
# as expected, we get an approximately linear increase in PE variance with alpha


### 2. How do we expect neural data to covary with prediction errors?
# X <- make_design_matrix(nTrials, PE)
# signal <- make_fMRI_signal(X, beta_PE=1)
# plot(signal[1:40,1], type='l', xlab='scan volume', ylab='Signal (A.U.)')

## Some example plots
# 1. Can we contrast the effects of two learning rates, one low and one high? Note that we turned off choice behavior here (so it's just a conditioning experiment, which allows for direct comparisons)
alpha1 <- 0.05
alpha2 <- 0.5
gamma=1
set.seed(2)
out1 <- simulate_RL_softmax(alpha=alpha1, beta=1, nTrials, pReward, rewardSensitivity=gamma, simulateChoice=FALSE)
set.seed(2)
out2 <- simulate_RL_softmax(alpha=alpha2, beta=1, nTrials, pReward, rewardSensitivity=gamma, simulateChoice=FALSE)

PE1 <- out1$PE
PE2 <- out2$PE
X1 <- make_design_matrix(nTrials, PE1)
X2 <- make_design_matrix(nTrials, PE2)
signal1 <- make_fMRI_signal(X1, beta_PE=1)  ## beta_PE = the 'phi' parameter in the paper
signal2 <- make_fMRI_signal(X2, beta_PE=1)




# Scenario 1 --------------------------------------------------------------
# Plotting settings
volumesPerTrial <- 4
nTrialsToPlot <- 13
nVolumes = nTrialsToPlot*volumesPerTrial
startVolume = 104    # plot somewhere after the initial learning phase
startTrial = startVolume/volumesPerTrial
ylims <- c(-3, 3)

palette(c("black", 'lightslateblue', 'dark orange'))


for(figtype in c('pdf', 'png')) {
 # if(figtype == 'pdf') pdf(file='./figures/scenario1.pdf', width=7, height=2.5)
 # if(figtype == 'png') png(file='./figures/scenario1.png', width=7, height=2.5, units='in', res=175)
  par(mfrow=c(1,3))
  par(mar=c(3,3.5,2,2), las=1, mgp=c(2,.5,0), oma=c(0,0,0,1))
  #expression(paste(alpha, " = ", c(1, 2, 3))
  plot(0,0, xlim=c(startVolume,startVolume+nVolumes), ylim=ylims, type='n', 
       xlab='Scan volume', ylab='Neural signal (a.u.)', bty='l', main=expression(bold(paste('A) Low ', alpha)))) #learning rate')
  abline(h=0, lty=2, col='grey')
  axis(side=4,at=c(-2,-1,0,1,2), labels=c(-2,'',0,'',2)); mtext('Prediction errors', side=4, cex=0.66, las=0,line=1)
  segments(x0=seq(startVolume, startVolume+nVolumes, 4),
           y0=rep(0, nTrialsToPlot),
           y1=PE1[startTrial:(nTrialsToPlot+startTrial),1], col=1, lwd=3)
  segments(x0=seq(startVolume, startVolume+nVolumes, 4),
           y0=rep(0, nTrialsToPlot),
           y1=PE1[startTrial:(nTrialsToPlot+startTrial),1], col=2, lwd=2)
  #lines(startVolume:(startVolume+nVolumes), signal1[(startVolume):(startVolume+nVolumes),1], col=1, lwd=3, lty='32')
  lines(startVolume:(startVolume+nVolumes), signal1[(startVolume):(startVolume+nVolumes),1], col=2, lwd=2, lty='31')
  legend('topleft', c('Prediction error', 'BOLD response'), bty='n', lwd=c(2,2), col=c(2,2), lty=c(1,2))
  
  
  #
  plot(0,0, xlim=c(startVolume,startVolume+nVolumes), ylim=ylims, type='n', 
       xlab='Scan volume', ylab='Neural signal (a.u.)', bty='l',  main=expression(bold(paste('B) High ', alpha))))
  abline(h=0, lty=2, col='grey')
  segments(x0=seq(startVolume, startVolume+nVolumes, 4),
           y0=rep(0, nTrialsToPlot),
           y1=PE2[startTrial:(nTrialsToPlot+startTrial),1], col=1, lwd=3)
  segments(x0=seq(startVolume, startVolume+nVolumes, 4),
           y0=rep(0, nTrialsToPlot),
           y1=PE2[startTrial:(nTrialsToPlot+startTrial),1], col=3, lwd=2)
  #lines(startVolume:(startVolume+nVolumes), signal2[(startVolume):(startVolume+nVolumes),1], col=1, lwd=3, lty='32')
  lines(startVolume:(startVolume+nVolumes), signal2[(startVolume):(startVolume+nVolumes),1], col=3, lwd=2, lty='31')
  legend('topleft', c('Prediction error', 'BOLD response'), bty='n', lwd=c(2,2), col=c(3,3), lty=c(1,2))
  axis(side=4,at=c(-2,-1,0,1,2), labels=c(-2,'',0,'',2)); mtext('Prediction errors', side=4, cex=0.66, las=0,line=1)
  
  
  #
  plot(0,0, xlim=c(startVolume,startVolume+nVolumes), ylim=ylims, type='n', xlab='Scan volume', ylab='Neural signal (a.u.)', bty='l', main='C) Comparison')
  abline(h=0, lty=2, col='grey')
  axis(side=4,at=c(-2,-1,0,1,2), labels=c(-2,'',0,'',2)); mtext('Prediction errors', side=4, cex=0.66, las=0,line=1)
  # Plot PEs of low alpha
  segments(x0=seq(startVolume, startVolume+nVolumes, 4)-.25,
           y0=rep(0, nTrialsToPlot),
           y1=PE1[startTrial:(nTrialsToPlot+startTrial),1], col=1, lwd=3)
  segments(x0=seq(startVolume, startVolume+nVolumes, 4)-.25,
           y0=rep(0, nTrialsToPlot),
           y1=PE1[startTrial:(nTrialsToPlot+startTrial),1], col=2, lwd=2)
  # and high alpha
  segments(x0=seq(startVolume, startVolume+nVolumes, 4)+.25,
           y0=rep(0, nTrialsToPlot),
           y1=PE2[startTrial:(nTrialsToPlot+startTrial),1], col=1, lwd=3)
  segments(x0=seq(startVolume, startVolume+nVolumes, 4)+.25,
           y0=rep(0, nTrialsToPlot),
           y1=PE2[startTrial:(nTrialsToPlot+startTrial),1], col=3, lwd=2)
  
  ## and BOLD signals
  lines(startVolume:(startVolume+nVolumes), signal1[(startVolume):(startVolume+nVolumes),1], col=2, lwd=2, lty='31')
  lines(startVolume:(startVolume+nVolumes), signal2[(startVolume):(startVolume+nVolumes),1], col=3, lwd=2, lty='31')
  legend('topleft', c(expression(paste('Low ', alpha)),
                      expression(paste('High ', alpha))), bty='n', lwd=c(2,2), col=c(2,3))
#  dev.off()
}




# Figure for revision: Only comparison in 1 panel -------------------------
# variances as a function of learning rate
alphas <- seq(0.005, 1, .025)
nReps <- 50
variances <- vector(mode='numeric', length=length(alphas))
for(i in 1:length(alphas)) {
  ## simulate multiple experiments to reduce the influence of simulation noise
  PEsthisalpha <- matrix(NA, nrow=nTrials, ncol=nReps)
  for(ii in 1:nReps) {
    out <- simulate_RL_softmax(alphas[i], beta, nTrials, pReward)
    PEsthisalpha[,ii] = out$PE
  }
  variances[i] <- mean(apply(PEsthisalpha,1,var))
}


# Scenario 1 Revision --------------------------------------------------------------

for(figtype in c('pdf', 'png')) {
  if(figtype == 'pdf') pdf(file='./figures/scenario1_revision.pdf', width=8, height=3)
  if(figtype == 'png') png(file='./figures/scenario1_revision.png', width=8, height=3, units='in', res=175)
#  par(mfrow=c(1,2), bty='l')
  layout(matrix(1:2, ncol=2), widths = c(.6, .4))
  par(mar=c(3,3.5,2,2.5), las=1, mgp=c(2,.5,0), oma=c(0,0,0,0), bty='l')
  volumesPerTrial <- 4
  nTrialsToPlot <- 20
  nVolumes = nTrialsToPlot*volumesPerTrial
  startVolume = 500    # plot somewhere after the initial learning phase
  startTrial = startVolume/volumesPerTrial
  
  plot(0,0, xlim=c(startVolume,startVolume+nVolumes), ylim=ylims, type='n', xlab='Time', ylab='Neural signal (a.u.)', main='A',xaxt='n')
  axis(1, at=pretty(seq(startVolume, startVolume+nVolumes)), label=NA)
  abline(h=0, lty=2, col='grey')
  axis(side=4,at=c(-2,-1,0,1,2), labels=c(-2,'',0,'',2)); mtext('Prediction errors', side=4, cex=par()$cex*par()$cex.axis, las=0,line=1.5)
  # Plot PEs of low alpha
  segments(x0=seq(startVolume, startVolume+nVolumes, 4)-.25,
           y0=rep(0, nTrialsToPlot),
           y1=PE1[startTrial:(nTrialsToPlot+startTrial),1], col=1, lwd=3)
  segments(x0=seq(startVolume, startVolume+nVolumes, 4)-.25,
           y0=rep(0, nTrialsToPlot),
           y1=PE1[startTrial:(nTrialsToPlot+startTrial),1], col=2, lwd=2)
  # and high alpha
  segments(x0=seq(startVolume, startVolume+nVolumes, 4)+.25,
           y0=rep(0, nTrialsToPlot),
           y1=PE2[startTrial:(nTrialsToPlot+startTrial),1], col=1, lwd=3)
  segments(x0=seq(startVolume, startVolume+nVolumes, 4)+.25,
           y0=rep(0, nTrialsToPlot),
           y1=PE2[startTrial:(nTrialsToPlot+startTrial),1], col=3, lwd=2)
  
  ## and BOLD signals
  lines(startVolume:(startVolume+nVolumes), signal1[(startVolume):(startVolume+nVolumes),1], col=2, lwd=2, lty='31')
  lines(startVolume:(startVolume+nVolumes), signal2[(startVolume):(startVolume+nVolumes),1], col=3, lwd=2, lty='31')
  legend('topleft', c(expression(paste('Low ', alpha)),
                      expression(paste('High ', alpha))), bty='n', lwd=c(2,2), col=c(2,3))
  
  ## variances
  par(mar=c(3,3.5,2,1))
  plot(alphas, variances, ylab='Prediction error variance', xlab=expression(paste('Learning rate ', alpha)), main='B', pch=4)
  abline(v=0.05,col=2,lwd=2)
  abline(v=0.5,col=3, lwd=2)
  dev.off()
}





# Scenario 7 --------------------------------------------------------------
### What if we erroneously assume the same learning rate for both participants?
# find PEs
updated1 <- emcAdapt::adapt.c.emc(out1$rewards, arguments=list(startValues=c(0,0), learningRates=matrix(.5, nrow=nTrials, ncol=2)))
updated2 <- emcAdapt::adapt.c.emc(out2$rewards, arguments=list(startValues=c(0,0), learningRates=matrix(.5, nrow=nTrials, ncol=2)))

PEs1_fit <- updated1$predictionErrors
PEs2_fit <- updated2$predictionErrors

X1_fit <- make_design_matrix(nTrials, apply(PEs1_fit,1,sum,na.rm=TRUE))
X2_fit <- make_design_matrix(nTrials, apply(PEs2_fit,1,sum,na.rm=TRUE))

par(mfrow=c(2,1))
plot(X1_fit[,1], xlim=c(0,40), lty=2, type='l', col=2)
lines(X2_fit[,1], xlim=c(0,40))
lines(signal1)
lines(signal2, col=2)

## we underestimate beta1, correctly find beta2
betas1_misspecified = fit_GLM(X1_fit, signal1)
betas2_misspecified = fit_GLM(X2_fit, signal2)

## if we correctly specify the model
updated1 <- emcAdapt::adapt.c.emc(out1$rewards, arguments=list(startValues=c(0,0), learningRates=matrix(alpha1, nrow=nTrials, ncol=2)))
updated2 <- emcAdapt::adapt.c.emc(out2$rewards, arguments=list(startValues=c(0,0), learningRates=matrix(alpha2, nrow=nTrials, ncol=2)))
PEs1_fit <- updated1$predictionErrors
PEs2_fit <- updated2$predictionErrors
X1_fit <- make_design_matrix(nTrials, apply(PEs1_fit,1,sum,na.rm=TRUE))
X2_fit <- make_design_matrix(nTrials, apply(PEs2_fit,1,sum,na.rm=TRUE))
betas1_correct = fit_GLM(X1_fit, signal1)
betas2_correct = fit_GLM(X2_fit, signal2)


# Scenario 7 plot ---------------------------------------------------------
for(figtype in c('pdf', 'png')) {
  if(figtype == 'pdf') pdf(file='./figures/scenario7.pdf', width=7, height=2.5)
  if(figtype == 'png') png(file='./figures/scenario7.png', width=7, height=2.5, units='in', res=175)
  
  l <- layout(matrix(c(1,1,2,2,3,4,5,6,1,1,2,2), nrow=3, byrow = TRUE), heights=c(.01, .9, .01))
  #layout.show(l)
  #par(mfrow=c())
  plot.new()
  mtext(expression(bold('A) Correct specification')), line=2.5, cex=.66*1.2)
  plot.new()
  mtext(expression(bold('B) Misspecification')), line=2.5, cex=.66*1.2)
  par(mar=c(3,4,3,.5), las=1, mgp=c(2,.5,0))
  barplot(c(.05, 0.5), ylab=expression(alpha), xlab='Participant', names.arg=c('A', 'B'))#, main='Correct specification')
  par(mar=c(3,3,3,1.5), las=1, mgp=c(2,.5,0))
  barplot(c(betas1_correct[1], betas2_correct[1]), ylab=expression(paste('Estimated ', phi)), xlab='Participant', names.arg=c('A', 'B'))
  
  par(mar=c(3,4,3,.5), las=1, mgp=c(2,.5,0))
  barplot(c(.5, 0.5), ylab=expression(alpha), xlab='Participant', names.arg=c('A', 'B'))#, main='Misspecification')
  par(mar=c(3,3,3,1.5), las=1, mgp=c(2,.5,0))
  barplot(c(betas1_misspecified[1], betas2_misspecified[1]), ylab=expression(paste('Estimated ', phi)), xlab='Participant', names.arg=c('A', 'B'), ylim=c(0,1))
  dev.off()
}

# Scenario 7 More noise, iterate across simulations --------------------------------------------------------------
# nIter = 10
# out <- data.frame(iter=1:nIter, misspecified_phi_1=NA, misspecified_phi_2=NA, correct_phi_1=NA, correct_phi_2=NA)

single_sim <- function(seed, sd_noise=10, nTrials=300,alpha1=0.05,alpha2=0.5, gamma=1) {
  set.seed(seed)
  out1 <- simulate_RL_softmax(alpha=alpha1, beta=1, nTrials=nTrials, pReward, rewardSensitivity=gamma, simulateChoice=TRUE)
  set.seed(seed)
  out2 <- simulate_RL_softmax(alpha=alpha2, beta=1, nTrials=nTrials, pReward, rewardSensitivity=gamma, simulateChoice=TRUE)
  
  PE1 <- out1$PE
  PE2 <- out2$PE
  X1 <- make_design_matrix(nTrials, PE1)
  X2 <- make_design_matrix(nTrials, PE2)
  
  signal1 <- make_fMRI_signal(X1, beta_PE=1, sd_noise = sd_noise)  ## beta_PE = the 'phi' parameter in the paper
  signal2 <- make_fMRI_signal(X2, beta_PE=1, sd_noise = sd_noise)
  
  # fit model with learning rate misspecified to 0.5
  updated1 <- emcAdapt::adapt.c.emc(out1$rewards, arguments=list(startValues=c(0,0), learningRates=matrix(.5, nrow=nTrials, ncol=2)))
  updated2 <- emcAdapt::adapt.c.emc(out2$rewards, arguments=list(startValues=c(0,0), learningRates=matrix(.5, nrow=nTrials, ncol=2)))
  
  PEs1_fit <- updated1$predictionErrors
  PEs2_fit <- updated2$predictionErrors
  
  X1_fit <- make_design_matrix(nTrials, apply(PEs1_fit,1,sum,na.rm=TRUE))
  X2_fit <- make_design_matrix(nTrials, apply(PEs2_fit,1,sum,na.rm=TRUE))
  
  betas1_misspecified = fit_GLM(X1_fit, signal1)
  betas2_misspecified = fit_GLM(X2_fit, signal2)
  
  ## if we correctly specify the model by estimating alpha
  optim1 <- DEoptim(ll_func, 
                  lower=c(0), 
                  upper=c(1), 
                  pnames='alpha', rewards=out1$rewards, choices=out1$choices, constants=c('beta'=1, 'gamma'=1))
  optim2 <- DEoptim(ll_func, 
                    lower=c(0), 
                    upper=c(1), 
                    pnames='alpha', rewards=out2$rewards, choices=out2$choices, constants=c('beta'=1, 'gamma'=1))
  updated1 <- emcAdapt::adapt.c.emc(out1$rewards, 
                                    arguments=list(startValues=c(0,0), learningRates=matrix(optim1$optim$bestmem[[1]], nrow=nTrials, ncol=2)))
  updated2 <- emcAdapt::adapt.c.emc(out2$rewards, 
                                    arguments=list(startValues=c(0,0), learningRates=matrix(optim2$optim$bestmem[[1]], nrow=nTrials, ncol=2)))
  #  updated1 <- emcAdapt::adapt.c.emc(out1$rewards, arguments=list(startValues=c(0,0), learningRates=matrix(alpha1, nrow=nTrials, ncol=2)))
  #updated2 <- emcAdapt::adapt.c.emc(out2$rewards, arguments=list(startValues=c(0,0), learningRates=matrix(alpha2, nrow=nTrials, ncol=2)))
  PEs1_fit <- updated1$predictionErrors
  PEs2_fit <- updated2$predictionErrors
  X1_fit <- make_design_matrix(nTrials, apply(PEs1_fit,1,sum,na.rm=TRUE))
  X2_fit <- make_design_matrix(nTrials, apply(PEs2_fit,1,sum,na.rm=TRUE))
  betas1_correct = fit_GLM(X1_fit, signal1)
  betas2_correct = fit_GLM(X2_fit, signal2)
  return(c(seed, betas1_correct[1], betas2_correct[1], betas1_misspecified[1], betas2_misspecified[1], alpha1=optim1$optim$bestmem[[1]], alpha2=optim2$optim$bestmem[[1]]))
}


out <- do.call(rbind, mclapply(1:400, single_sim, sd_noise=10, nTrials=300, mc.cores = 8))  # NB: sd=10 and contrast=1, so CNR=1/10 =0.1, so *half* the CNR of Miletic et al. 2023
means <- apply(out,2,mean)[2:ncol(out)]
SEs <- apply(out, 2, function(x) sd(x)/sqrt(length(x)))[2:ncol(out)]

library(gplots)
pdf(file='./figures/scenario7_revision.pdf', width=7, height=2.5)
l <- layout(matrix(c(1,1,2,2,3,4,5,6,1,1,2,2), nrow=3, byrow = TRUE), heights=c(.01, .9, .01))
plot.new()
mtext(expression(bold('A) Correct specification')), line=2.5, cex=.66*1.2)
plot.new()
mtext(expression(bold('B) Misspecification')), line=2.5, cex=.66*1.2)
par(mar=c(3,4,3,.5), las=1, mgp=c(2,.5,0))
gplots::barplot2(height=means[5:6], ci.l=means[5:6]-SEs[5:6], ci.u=means[5:6]+SEs[5:6],plot.ci = TRUE, ylab=expression(paste('Estimated ', phi)), xlab='Participant', names.arg=c('A', 'B'), ylim=c(0,1), col = 2:3)
# barplot(c(.05, 0.5), ylab=expression(alpha), xlab='Participant', names.arg=c('A', 'B'), col = 2:3)#, main='Correct specification')
par(mar=c(3,3,3,1.5), las=1, mgp=c(2,.5,0))
gplots::barplot2(height=means[1:2], ci.l=means[1:2]-SEs[1:2], ci.u=means[1:2]+SEs[1:2],plot.ci = TRUE, ylab=expression(paste('Estimated ', phi)), xlab='Participant', names.arg=c('A', 'B'), ylim=c(0,1), col = 2:3)

par(mar=c(3,4,3,.5), las=1, mgp=c(2,.5,0))
barplot(c(.5, 0.5), ylab=expression(alpha), xlab='Participant', names.arg=c('A', 'B'), col = 2:3)#, main='Misspecification')
par(mar=c(3,3,3,1.5), las=1, mgp=c(2,.5,0))
gplots::barplot2(height=means[1:2+2], ci.l=means[1:2+2]-SEs[1:2+2], ci.u=means[1:2+2]+SEs[1:2+2],plot.ci = TRUE, ylab=expression(paste('Estimated ', phi)), xlab='Participant', names.arg=c('A', 'B'), ylim=c(0,1), col = 2:3)
dev.off()

# 
# l <- layout(matrix(c(1,1,2,2,3,4,5,6,1,1,2,2), nrow=3, byrow = TRUE), heights=c(.01, .9, .01))
# #layout.show(l)
# #par(mfrow=c())
# plot.new()
# mtext(expression(bold('A) Correct specification')), line=2.5, cex=.66*1.2)
# plot.new()
# mtext(expression(bold('B) Misspecification')), line=2.5, cex=.66*1.2)
# par(mar=c(3,4,3,.5), las=1, mgp=c(2,.5,0))
# barplot(c(.05, 0.5), ylab=expression(alpha), xlab='Participant', names.arg=c('A', 'B'))#, main='Correct specification')
# par(mar=c(3,3,3,1.5), las=1, mgp=c(2,.5,0))
# barplot(c(betas1_correct[1], betas2_correct[1]), ylab=expression(paste('Estimated ', phi)), xlab='Participant', names.arg=c('A', 'B'))
# 
# par(mar=c(3,4,3,.5), las=1, mgp=c(2,.5,0))
# barplot(c(.5, 0.5), ylab=expression(alpha), xlab='Participant', names.arg=c('A', 'B'))#, main='Misspecification')
# par(mar=c(3,3,3,1.5), las=1, mgp=c(2,.5,0))
# barplot(c(betas1_misspecified[1], betas2_misspecified[1]), ylab=expression(paste('Estimated ', phi)), xlab='Participant', names.arg=c('A', 'B'), ylim=c(0,1))




# Scenario 2 plot ---------------------------------------------------------
signal1 <- make_fMRI_signal(X1, beta_PE=1.5)    # use same design matrix, just add differneces in phi
signal2 <- make_fMRI_signal(X2, beta_PE=.5)

for(figtype in c('pdf', 'png')) {
  if(figtype == 'pdf') pdf(file='./figures/scenario2.pdf', width=7, height=2.5)
  if(figtype == 'png') png(file='./figures/scenario2.png', width=7, height=2.5, units='in', res=175)
  par(mfrow=c(1,3))
  par(mar=c(3,3.5,2,2), las=1, mgp=c(2,.5,0),oma=c(0,0,0,1))
  plot(0,0, xlim=c(startVolume,startVolume+nVolumes), ylim=ylims, type='n', xlab='Scan volume', ylab='Neural signal (a.u.)', bty='l', main=expression(bold(paste('A) Low ', alpha, ', high ', phi)))) #learning rate, high phi')
  axis(side=4,at=c(-2,-1,0,1,2), labels=c(-2,'',0,'',2)); mtext('Prediction errors', side=4, cex=0.66, las=0,line=1)
  abline(h=0, lty=2, col='grey')
  segments(x0=seq(startVolume, startVolume+nVolumes, 4),
           y0=rep(0, nTrialsToPlot),
           y1=PE1[startTrial:(nTrialsToPlot+startTrial),1], col=1, lwd=3)
  segments(x0=seq(startVolume, startVolume+nVolumes, 4),
           y0=rep(0, nTrialsToPlot),
           y1=PE1[startTrial:(nTrialsToPlot+startTrial),1], col=2, lwd=2)
  lines(startVolume:(startVolume+nVolumes), signal1[(startVolume):(startVolume+nVolumes),1], col=2, lwd=2, lty='31')
  legend('topleft', c('Prediction error', 'BOLD response'), bty='n', lwd=c(2,2), col=c(2,2), lty=c(1,2))
  
  
  #
  plot(0,0, xlim=c(startVolume,startVolume+nVolumes), ylim=ylims, type='n', xlab='Scan volume', ylab='Neural signal (a.u.)', bty='l', main=expression(bold(paste('B) High ', alpha, ', low ', phi)))) #learning rate, high phi')
  axis(side=4,at=c(-2,-1,0,1,2), labels=c(-2,'',0,'',2)); mtext('Prediction errors', side=4, cex=0.66, las=0,line=1)
  abline(h=0, lty=2, col='grey')
  segments(x0=seq(startVolume, startVolume+nVolumes, 4),
           y0=rep(0, nTrialsToPlot),
           y1=PE2[startTrial:(nTrialsToPlot+startTrial),1], col=1, lwd=3)
  segments(x0=seq(startVolume, startVolume+nVolumes, 4),
           y0=rep(0, nTrialsToPlot),
           y1=PE2[startTrial:(nTrialsToPlot+startTrial),1], col=3, lwd=2)
  lines(startVolume:(startVolume+nVolumes), signal2[(startVolume):(startVolume+nVolumes),1], col=3, lwd=2, lty='31')
  legend('topleft', c('Prediction error', 'BOLD response'), bty='n', lwd=c(2,2), col=c(3,3), lty=c(1,2))
  
  
  #
  plot(0,0, xlim=c(startVolume,startVolume+nVolumes), ylim=ylims, type='n', xlab='Scan volume', ylab='Neural signal (a.u.)', bty='l', main='C) Comparison')
  axis(side=4,at=c(-2,-1,0,1,2), labels=c(-2,'',0,'',2)); mtext('Prediction errors', side=4, cex=0.66, las=0,line=1)
  abline(h=0, lty=2, col='grey')
  # Plot PEs of low alpha
  segments(x0=seq(startVolume, startVolume+nVolumes, 4)-.25,
           y0=rep(0, nTrialsToPlot),
           y1=PE1[startTrial:(nTrialsToPlot+startTrial),1], col=1, lwd=3)
  segments(x0=seq(startVolume, startVolume+nVolumes, 4)-.25,
           y0=rep(0, nTrialsToPlot),
           y1=PE1[startTrial:(nTrialsToPlot+startTrial),1], col=2, lwd=2)
  # and high alpha
  segments(x0=seq(startVolume, startVolume+nVolumes, 4)+.25,
           y0=rep(0, nTrialsToPlot),
           y1=PE2[startTrial:(nTrialsToPlot+startTrial),1], col=1, lwd=3)
  segments(x0=seq(startVolume, startVolume+nVolumes, 4)+.25,
           y0=rep(0, nTrialsToPlot),
           y1=PE2[startTrial:(nTrialsToPlot+startTrial),1], col=3, lwd=2)
  
  ## and BOLD signals
  lines(startVolume:(startVolume+nVolumes), signal1[(startVolume):(startVolume+nVolumes),1], col=2, lwd=2, lty='31')
  lines(startVolume:(startVolume+nVolumes), signal2[(startVolume):(startVolume+nVolumes),1], col=3, lwd=2, lty='31')
  
  legend('topleft', c(expression(paste('Low ', alpha, ', high ', phi)),
                      expression(paste('High ', alpha, ', low ', phi))), bty='n', lwd=c(2,2), col=c(2,3))
  dev.off()
}




## Same here, newer figure
#layout(matrix(1:2, ncol=2), widths = c(.6, .4))
pdf('./figures/scenario2_revision.pdf', width=5, height=3)
par(mar=c(3,3.5,2,2.5), las=1, mgp=c(2,.5,0), oma=c(0,0,0,0), bty='l')
volumesPerTrial <- 4
nTrialsToPlot <- 20
nVolumes = nTrialsToPlot*volumesPerTrial
startVolume = 500    # plot somewhere after the initial learning phase
startTrial = startVolume/volumesPerTrial

plot(0,0, xlim=c(startVolume,startVolume+nVolumes), ylim=ylims, type='n', xlab='Time', ylab='Neural signal (a.u.)', bty='l', main='', xaxt='n')
axis(1, at=pretty(seq(startVolume, startVolume+nVolumes)), label=NA)
axis(side=4,at=c(-2,-1,0,1,2), labels=c(-2,'',0,'',2)); mtext('Prediction errors', side=4, cex=par()$cex*par()$cex.axis, las=0,line=1.5)
abline(h=0, lty=2, col='grey')
# Plot PEs of low alpha
segments(x0=seq(startVolume, startVolume+nVolumes, 4)-.25,
         y0=rep(0, nTrialsToPlot),
         y1=PE1[startTrial:(nTrialsToPlot+startTrial),1], col=1, lwd=3)
segments(x0=seq(startVolume, startVolume+nVolumes, 4)-.25,
         y0=rep(0, nTrialsToPlot),
         y1=PE1[startTrial:(nTrialsToPlot+startTrial),1], col=2, lwd=2)
# and high alpha
segments(x0=seq(startVolume, startVolume+nVolumes, 4)+.25,
         y0=rep(0, nTrialsToPlot),
         y1=PE2[startTrial:(nTrialsToPlot+startTrial),1], col=1, lwd=3)
segments(x0=seq(startVolume, startVolume+nVolumes, 4)+.25,
         y0=rep(0, nTrialsToPlot),
         y1=PE2[startTrial:(nTrialsToPlot+startTrial),1], col=3, lwd=2)

## and BOLD signals
lines(startVolume:(startVolume+nVolumes), signal1[(startVolume):(startVolume+nVolumes),1], col=2, lwd=2, lty='31')
lines(startVolume:(startVolume+nVolumes), signal2[(startVolume):(startVolume+nVolumes),1], col=3, lwd=2, lty='31')

legend('topleft', c(expression(paste('Low ', alpha, ', high ', phi)),
                    expression(paste('High ', alpha, ', low ', phi))), bty='n', lwd=c(2,2), col=c(2,3))
dev.off()

## variances
#par(mar=c(3,3.5,2,1))
#plot(alphas, variances, ylab='Prediction error variance', xlab=expression(paste('Learning rate ', alpha)), main='B', pch=4)
#abline(v=0.05,col=2,lwd=2)
#abline(v=0.5,col=3, lwd=2)
