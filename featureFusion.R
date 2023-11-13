## Farzad Zandi, 2023.
# Fusing extracted features.

rm(list=ls())

## Read AD Features.
ADna <- read.csv("/Extracted Features/NA/AD.csv")
ADnb <- read.csv("/Extracted Features/NB/AD.csv")
ADpa <- read.csv("/Extracted Features/PA/AD.csv")
ADpb <- read.csv("/Extracted Features/PB/AD.csv")
ADn = as.data.frame(c(ADna[,-1], ADnb[,-1]))
ADp = as.data.frame(c(ADpa[,-1], ADpb[,-1]))
AD = rbind(ADn, ADp)

## Read BLOSUM Features.
BLOSUMna <- read.csv("/Extracted Features/NA/BLOSUM.csv")
BLOSUMnb <- read.csv("/Extracted Features/NB/BLOSUM.csv")
BLOSUMpa <- read.csv("/Extracted Features/PA/BLOSUM.csv")
BLOSUMpb <- read.csv("/Extracted Features/PB/BLOSUM.csv")
BLOSUMn = as.data.frame(c(BLOSUMna[,-1], BLOSUMnb[,-1]))
BLOSUMp = as.data.frame(c(BLOSUMpa[,-1], BLOSUMpb[,-1]))
BLOSUM = rbind(BLOSUMn, BLOSUMp)

## Read CT Features.
CTna <- read.csv("/Extracted Features/NA/CT.csv")
CTnb <- read.csv("/Extracted Features/NB/CT.csv")
CTpa <- read.csv("/Extracted Features/PA/CT.csv")
CTpb <- read.csv("/Extracted Features/PB/CT.csv")
CTn = as.data.frame(c(CTna[,-1], CTnb[,-1]))
CTp = as.data.frame(c(CTpa[,-1], CTpb[,-1]))
CT = rbind(CTn, CTp)

## Read C-T-D Features.
CTDna <- read.csv("/Extracted Features/NA/C-T-D.csv")
CTDnb <- read.csv("/Extracted Features/NB/C-T-D.csv")
CTDpa <- read.csv("/Extracted Features/PA/C-T-D.csv")
CTDpb <- read.csv("/Extracted Features/PB/C-T-D.csv")
CTDn = as.data.frame(c(CTDna[,-1], CTDnb[,-1]))
CTDp = as.data.frame(c(CTDpa[,-1], CTDpb[,-1]))
CTD = rbind(CTDn, CTDp)

## Read DC Features.
DCna <- read.csv("/Extracted Features/NA/DC.csv")
DCnb <- read.csv("/Extracted Features/NB/DC.csv")
DCpa <- read.csv("/Extracted Features/PA/DC.csv")
DCpb <- read.csv("/Extracted Features/PB/DC.csv")
DCn = as.data.frame(c(DCna[,-1], DCnb[,-1]))
DCp = as.data.frame(c(DCpa[,-1], DCpb[,-1]))
DC = rbind(DCn, DCp)

## Read DDE Features.
DDEna <- read.csv("/Extracted Features/NA/DDE.csv")
DDEnb <- read.csv("/Extracted Features/NB/DDE.csv")
DDEpa <- read.csv("/Extracted Features/PA/DDE.csv")
DDEpb <- read.csv("/Extracted Features/PB/DDE.csv")
DDEn = as.data.frame(c(DDEna[,-1], DDEnb[,-1]))
DDEp = as.data.frame(c(DDEpa[,-1], DDEpb[,-1]))
DDE = rbind(DDEn, DDEp)

## Read PseAAC Features.
PseAACna <- read.csv("/Extracted Features/NA/PseAAC.csv")
PseAACnb <- read.csv("/Extracted Features/NB/PseAAC.csv")
PseAACpa <- read.csv("/Extracted Features/PA/PseAAC.csv")
PseAACpb <- read.csv("/Extracted Features/PB/PseAAC.csv")
PseAACn = as.data.frame(c(PseAACna[,-1], PseAACnb[,-1]))
PseAACp = as.data.frame(c(PseAACpa[,-1], PseAACpb[,-1]))
PseAAC = rbind(PseAACn, PseAACp)

## Read QSO Features.
QSOna <- read.csv("/Extracted Features/NA/QSO.csv")
QSOnb <- read.csv("/Extracted Features/NB/QSO.csv")
QSOpa <- read.csv("/Extracted Features/PA/QSO.csv")
QSOpb <- read.csv("/Extracted Features/PB/QSO.csv")
QSOn = as.data.frame(c(QSOna[,-1], QSOnb[,-1]))
QSOp = as.data.frame(c(QSOpa[,-1], QSOpb[,-1]))
QSO = rbind(QSOn, QSOp)

## Fusion ALL Features.
fusion = as.data.frame(c(AD, BLOSUM, CT, CTD, DC, DDE, PseAAC, QSO))

## Save All Features.
write.csv(AD, "/Extracted Features/Fusion/AD.csv")
write.csv(BLOSUM, "/Extracted Features/Fusion/BLOSUM.csv")
write.csv(CT, "/Extracted Features/Fusion/CT.csv")
write.csv(CTD, "/Extracted Features/Fusion/C-T-D.csv")
write.csv(DC, "/Extracted Features/Fusion/DC.csv")
write.csv(DDE, "/Extracted Features/Fusion/DDE.csv")
write.csv(PseAAC, "/Extracted Features/Fusion/PseAAC.csv")
write.csv(QSO, "/Extracted Features/Fusion/QSO.csv")
write.csv(fusion, "/Extracted Features/Fusion/Fusion.csv")



