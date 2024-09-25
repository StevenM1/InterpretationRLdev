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

## Assume a RL process with the following parameters
nTrials = 1e3
out = simulate_RL_softmax(alpha=0.1, beta=1, nTrials=nTrials, pReward=c(.8,.2))
PEs <- out$PE  # get the prediction errors

# Generate design matrix with duration of 10 ms. We'll use this design matrix to fit a GLM to fMRI data that was
# simulated with a longer duration
X_short <- make_design_matrix(nTrials, PEs, TR=1, duration = 0.01, oversampling=100)

# To generate a DM and fMRI signal with a longer duration...
#X_long <- make_design_matrix(nTrials, PEs, TR=1, duration=0.1)
#fmri_long <- make_fMRI_signal(X_long, beta_PE=1)
#fit_GLM(X_short, fmri_long)

# Do this repeatedly for 100 varying 'true' BOLD signal durations
true_durations = seq(0.01, 1, length.out=100)
neuralCoding <- mclapply(true_durations, function(x) {
  X_long <- make_design_matrix(nTrials, PEs, TR=1, duration=x, oversampling=100)
  fmri_long <- make_fMRI_signal(X_long, beta_PE=1)
  return(fit_GLM(X_short, fmri_long)[1,1])
}, mc.cores=8)
neuralCoding <- unlist(neuralCoding)

plot(true_durations,neuralCoding,type="l",xlim=c(0,0.1),ylim=c(min(neuralCoding)-.05,max(neuralCoding)+.05),ylab="Phi",xlab="True Duration")
arrows(x0=0.01, y0=-15, y1=-5, xpd=TRUE, length=0.1,col=2,lwd=2)
text(x=0.01, y=-20, labels='Assumed duration', xpd=TRUE, col=2)


## Plot
frame_times <- seq(0, 22, .1)
events <- data.frame('onset'=c(0,0,0,0,0,0), 'trial_type'=c('durp1mod1', 'durp2mod1', 'durp1mod2',
                                                            'dur1mod1', 'dur2mod1', 'dur1mod2'), 
                     'duration'=c(0.1, 0.2, 0.1, 1, 2, 1), 'modulation'=c(1,1,2, 1,1,2))
X <- as.matrix(make_fmri_design_matrix(frame_times=frame_times, events=events))  # design matrix

pdf(file='./figures/scenario4.pdf', width=7, height=2.5)
par(mfrow=c(1,3))
par(mar=c(3,3.5,2,.5), las=1, mgp=c(2,.5,0))
plot(X[,4], type='l', ylim=range(X[,4:6]), lwd=2, bty='l', ylab='BOLD response (a.u.)', yaxt='n', xaxt='n', xlab='Time (s)', main='A) Duration ± 0.1 s')
abline(h=0, lty=2, col='grey')
axis(side=1, at=c(0, 50, 100, 150, 200), labels=c(0, 5, 10, 15, 20))
lines(X[,5], col=2, lwd=2)
lines(X[,6], col=3, lwd=2)
legend('topright', c(expression(paste('Duration = 0.1 s, ', phi, ' = 1')),
                     expression(paste('Duration = 0.2 s, ', phi, ' = 1')),
                     expression(paste('Duration = 0.1 s, ', phi, ' = 2'))), col=c(1,2,3), lwd=c(2,2,2), bty='n', cex=.8)

plot(X[,1], type='l', ylim=range(X[,1:3]), lwd=2, bty='l', ylab='BOLD response (a.u.)', yaxt='n', xaxt='n', xlab='Time (s)', main='B) Duration ± 1 s')
abline(h=0, lty=2, col='grey')
axis(side=1, at=c(0, 50, 100, 150, 200), labels=c(0, 5, 10, 15, 20))
lines(X[,2], col=2, lwd=2)
lines(X[,3], col=3, lwd=2)
legend('topright', c(expression(paste('Duration = 1 s, ', phi, ' = 1')),
                     expression(paste('Duration = 2 s, ', phi, ' = 1')),
                     expression(paste('Duration = 1 s, ', phi, ' = 2'))), col=c(1,2,3), lwd=c(2,2,2), bty='n', cex=.8)

#
plot(true_durations,neuralCoding,type="l",ylim=c(min(neuralCoding)-.05,max(neuralCoding)+.05),ylab=expression(phi),xlab="True duration", bty='l', lwd=2, main='C) Misspecified duration')
axis(side=1, at=0.01, labels='', tck=0.035,col=2,lwd=2)
arrows(x0=0.04, x1=0.2, y0=-1.25, y1=2.5, xpd=TRUE, length=0.05,col=2,lwd=2, code=1)
text(x=0.2, y=2.6, labels=expression(bold('Assumed duration')), xpd=TRUE, col=2, pos=4)
dev.off()