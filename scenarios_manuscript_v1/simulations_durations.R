rm(list=ls())
library(DEoptim)
library(parallel)
library(pracma)


## custom code
source('./fMRI.R') ## <- some code to generate fMRI design matrices and the like
source('./simulation_functions.R')


# 1. Can we contrast the effects of two different durations, one low and one high? Note that we turned off choice behavior here (so it's just a conditioning experiment, which allows for direct comparisons)
pReward <- c(.8, .2)
alpha1 <- .1
nTrials <- 1e3
duration1 <- 0.1   # 100 ms
duration2 <- 0.5   # 500 ms
gamma=1
set.seed(2)
out1 <- simulate_RL_softmax(alpha=alpha1, beta=1, nTrials, pReward, rewardSensitivity=gamma, simulateChoice=FALSE)
set.seed(2)
out2 <- simulate_RL_softmax(alpha=alpha1, beta=1, nTrials, pReward, rewardSensitivity=gamma, simulateChoice=FALSE)

PE1 <- out1$PE
PE2 <- out2$PE

## NB: PE1 == PE2, so we assume these two participants are exactly equal in their behavior
X1 <- make_design_matrix(nTrials, PE1, duration = duration1)
X2 <- make_design_matrix(nTrials, PE2, duration = duration2)
signal1 <- make_fMRI_signal(X1, beta_PE=1)  ## beta_PE = the 'phi' parameter in the paper
signal2 <- make_fMRI_signal(X2, beta_PE=1)

# Plotting
volumesPerTrial <- 4
nTrialsToPlot <- 20
nVolumes = nTrialsToPlot*volumesPerTrial
startVolume = 100    # plot somewhere after the initial learning phase
startTrial = startVolume/volumesPerTrial
ylims <- c(-20, 20)


palette(c("black", 'lightslateblue', 'dark orange'))

layout(matrix(c(1,2,3,3), nrow=2, ncol=2, byrow=TRUE))
par(mar=c(3,3.5,2,.5), las=1, mgp=c(2,.5,0))
plot(0,0, xlim=c(startVolume,startVolume+nVolumes), ylim=ylims, type='n', xlab='Scan volume', ylab='Neural signal (a.u.)', bty='l', main='A) Short duration')
abline(h=0, lty=2, col='grey')
segments(x0=seq(startVolume, startVolume+nVolumes, 4),
         y0=rep(0, nTrialsToPlot),
         y1=PE1[startTrial:(nTrialsToPlot+startTrial),1], col=2, lwd=2)
lines(startVolume:(startVolume+nVolumes), signal1[(startVolume):(startVolume+nVolumes),1], col=1, lwd=2)
legend('topright', c('Prediction error', 'BOLD response'), bty='n', lwd=c(2,2), col=c(2,1))


#
plot(0,0, xlim=c(startVolume,startVolume+nVolumes), ylim=ylims, type='n', xlab='Scan volume', ylab='Neural signal (a.u.)', bty='l', main='B) Long duration')
abline(h=0, lty=2, col='grey')
segments(x0=seq(startVolume, startVolume+nVolumes, 4),
         y0=rep(0, nTrialsToPlot),
         y1=PE2[startTrial:(nTrialsToPlot+startTrial),1], col=3, lwd=2)
lines(startVolume:(startVolume+nVolumes), signal2[(startVolume):(startVolume+nVolumes),1], col=1, lwd=2)
legend('topright', c('Prediction error', 'BOLD response'), bty='n', lwd=c(2,2), col=c(3,1))


#
plot(0,0, xlim=c(startVolume,startVolume+nVolumes), ylim=ylims, type='n', xlab='Scan volume', ylab='Neural signal (a.u.)', bty='l', main='C) Comparison')
abline(h=0, lty=2, col='grey')
# Plot PEs of low alpha
segments(x0=seq(startVolume, startVolume+nVolumes, 4)-.25,
         y0=rep(0, nTrialsToPlot),
         y1=PE1[startTrial:(nTrialsToPlot+startTrial),1], col=2, lwd=2)
# and high alpha
segments(x0=seq(startVolume, startVolume+nVolumes, 4)+.25,
         y0=rep(0, nTrialsToPlot),
         y1=PE2[startTrial:(nTrialsToPlot+startTrial),1], col=3, lwd=2)

## and BOLD signals
lines(startVolume:(startVolume+nVolumes), signal1[(startVolume):(startVolume+nVolumes),1], col=2, lwd=2, lty=2)

lines(startVolume:(startVolume+nVolumes), signal2[(startVolume):(startVolume+nVolumes),1], col=3, lwd=2, lty=2)

legend('topright', c('Short duration', 'Long duration'), bty='n', lwd=c(2,2), col=c(2,3))

# subplot( 
#   #  plot(0, col=2, pch='.', mgp=c(1,0.4,0), xlab='', ylab='', cex.axis=0.5), 
#   barplot(height=c(var(PE1), var(PE2)), col=c(2,3), main='PE variance'),
#   x=grconvertX(c(0.025,.155), from='npc'),
#   y=grconvertY(c(0.6,1), from='npc'),
#   type='fig', pars=list( mar=c(1.5,1.5,1,0)+0.1, mgp=c(2,1,0), cex.main=.8, cex=.6) )
# 
# subplot( 
#   #  plot(0, col=2, pch='.', mgp=c(1,0.4,0), xlab='', ylab='', cex.axis=0.5), 
#   barplot(height=c(var(signal1), var(signal2)), col=c(2,3), main='BOLD variance'),
#   x=grconvertX(c(0.175+.05,.175+.05+0.13), from='npc'),
#   y=grconvertY(c(0.6,1), from='npc'),
#   type='fig', pars=list( mar=c(1.5,1.5,1,0)+0.1, mgp=c(2,1,0), cex.main=.8, cex=.6) )



## Simulate effect of assuming a too short duration
nTrials = 1e3
out = simulate_RL_softmax(0.1, 1, nTrials, c(.8,.2))
# Test for differences in neural coding parameter
PEs <- out$PE

X_short <- make_design_matrix(nTrials, PEs, TR=1, duration = 0.01, oversampling=100)
X_long <- make_design_matrix(nTrials, PEs, TR=1, duration=0.1)
fmri_long <- make_fMRI_signal(X_long, beta_PE=1)
fit_GLM(X_short, fmri_long)

#neuralCoding = numeric()
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



#### Plot of BOLD responses for multiple durations
frame_times <- seq(0, 22, .1)
events <- data.frame('onset'=c(0,0,0,0,0,0), 'trial_type'=c('durp1mod1', 'durp2mod1', 'durp1mod2',
                                                         'dur1mod1', 'dur2mod1', 'dur1mod2'), 
                     'duration'=c(0.1, 0.2, 0.1, 1, 2, 1), 'modulation'=c(1,1,2, 1,1,2))
X <- as.matrix(make_fmri_design_matrix(frame_times=frame_times, events=events))  # design matrix

for(figtype in c('pdf', 'png')) {
  if(figtype == 'pdf') pdf(file='./figures/scenario4.pdf', width=7, height=2.5)
  if(figtype == 'png') png(file='./figures/scenario4.png', width=7, height=2.5, units='in', res=175)
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
  #abline(v=0.01,lty=2,lwd=2,col=2)
  axis(side=1, at=0.01, labels='', tck=0.035,col=2,lwd=2)
  arrows(x0=0.04, x1=0.2, y0=-1.25, y1=2.5, xpd=TRUE, length=0.05,col=2,lwd=2, code=1)
  text(x=0.2, y=2.6, labels=expression(bold('Assumed duration')), xpd=TRUE, col=2, pos=4)
#  legend(-.05,90,text.col=2,bty="n",legend="Assumed duration", cex=.8)
  dev.off()
}


