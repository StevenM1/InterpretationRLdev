## Scenarios 1, 2, 7
rm(list=ls())
library(DEoptim)
library(parallel)
library(pracma)

## custom code
library(emcAdapt)  ## <- package for fast RL updating, using some C-code, attached
source('./fMRI.R') ## <- some code to generate fMRI design matrices and the like
source('./simulation_functions.R')
palette(c("black", 'lightslateblue', 'dark orange'))


nTrials <- 1e3
pReward <- c(0.8, 0.2)
# Scenario 1: Low vs high learning rate -----------------------------------
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


## For panel B: Variance as a function of learning rate
alphas <- seq(0.005, 1, .025)
nReps <- 50
variances <- vector(mode='numeric', length=length(alphas))
for(i in 1:length(alphas)) {
  ## simulate multiple experiments to reduce the influence of simulation noise
  PEsthisalpha <- matrix(NA, nrow=nTrials, ncol=nReps)
  for(ii in 1:nReps) {
    out <- simulate_RL_softmax(alphas[i], beta=1, nTrials, pReward)
    PEsthisalpha[,ii] = out$PE
  }
  variances[i] <- mean(apply(PEsthisalpha,1,var))
}


# Plotting settings
volumesPerTrial <- 4
nTrialsToPlot <- 13
nVolumes = nTrialsToPlot*volumesPerTrial
startVolume = 104    # plot somewhere after the initial learning phase
startTrial = startVolume/volumesPerTrial
ylims <- c(-3, 3)


# Plot
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



# Scenario 2: Add variability in phi --------------------------------------------------------------
signal1 <- make_fMRI_signal(X1, beta_PE=2)    # use same design matrix as above, just add differences in phi
signal2 <- make_fMRI_signal(X2, beta_PE=.5)

#dev.off()
volumesPerTrial <- 4
nTrialsToPlot <- 20
nVolumes = nTrialsToPlot*volumesPerTrial
startVolume = 500    # plot somewhere after the initial learning phase
startTrial = startVolume/volumesPerTrial

pdf('./figures/scenario3_revision.pdf', width=5, height=3)
par(mar=c(3,3.5,2,2.5), las=1, mgp=c(2,.5,0), oma=c(0,0,0,0), bty='l')
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




# Scenario 7 --------------------------------------------------------------
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
  # updated1 <- emcAdapt::adapt.c.emc(out1$rewards, arguments=list(startValues=c(0,0), learningRates=matrix(alpha1, nrow=nTrials, ncol=2)))
  # updated2 <- emcAdapt::adapt.c.emc(out2$rewards, arguments=list(startValues=c(0,0), learningRates=matrix(alpha2, nrow=nTrials, ncol=2)))
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
gplots::barplot2(height=means[5:6], ci.l=means[5:6]-SEs[5:6], ci.u=means[5:6]+SEs[5:6],plot.ci = TRUE, ylab=expression(paste('Estimated ', alpha)), xlab='Participant', names.arg=c('A', 'B'), ylim=c(0,.55), col = 2:3)
# barplot(c(.05, 0.5), ylab=expression(alpha), xlab='Participant', names.arg=c('A', 'B'), col = 2:3)#, main='Correct specification')
par(mar=c(3,3,3,1.5), las=1, mgp=c(2,.5,0))
gplots::barplot2(height=means[1:2], ci.l=means[1:2]-SEs[1:2], ci.u=means[1:2]+SEs[1:2],plot.ci = TRUE, ylab=expression(paste('Estimated ', phi)), xlab='Participant', names.arg=c('A', 'B'), ylim=c(0,1), col = 2:3)

par(mar=c(3,4,3,.5), las=1, mgp=c(2,.5,0))
barplot(c(.5, 0.5), ylab=expression(alpha), xlab='Participant', names.arg=c('A', 'B'), col = 2:3)#, main='Misspecification')
par(mar=c(3,3,3,1.5), las=1, mgp=c(2,.5,0))
gplots::barplot2(height=means[1:2+2], ci.l=means[1:2+2]-SEs[1:2+2], ci.u=means[1:2+2]+SEs[1:2+2],plot.ci = TRUE, ylab=expression(paste('Estimated ', phi)), xlab='Participant', names.arg=c('A', 'B'), ylim=c(0,1), col = 2:3)
dev.off()

