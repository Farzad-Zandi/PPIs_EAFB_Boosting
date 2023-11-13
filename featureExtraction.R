## Farzad Zandi, 2023.
# Extracting Protein Features.

rm(list=ls())
library(protr)
library(ftrCOOL)
library(R.matlab)

proteinPA <- readMat("/P_proteinA.mat")
proteinPA <- proteinPA$P.proteinA
dc = c()
dde = c()
qso = c()
ad = c()
ctd = c()
ct = c()
pseaac = c()
blosum = c()
for (i in 1:dim(proteinPA)[1])
{
  seq <- as.data.frame(proteinPA[i])
  seq <- seq$structure  
  out <- extractDC(seq) # Extract Dipeptide Composition Features.
  dc <- rbind(dc, out)
  out <- DDE(seq) # Extract Dipeptide Deviation from Expected Mean Features.
  dde <- rbind(dde, out)
  out = extractQSO(seq, nlag = 3, w = 0.1) # Extract Quasi Sequence Order Features.
  qso <- rbind(qso, out)
  geary <- extractGeary(seq, props = c("CIDH920105", "BHAR880101", "CHAM820101", # Extract Geary Autocorrelation Features.
                                       "CHAM820102", "CHOC760101", "BIGC670101", "CHAM810101", 
                                       "DAYM780201"), nlag = 3, customprops = NULL)
  moran <- extractMoran(seq, props = c("CIDH920105", "BHAR880101", "CHAM820101", # Extract Moran Autocorrelation Features.
                                       "CHAM820102", "CHOC760101", "BIGC670101", "CHAM810101", 
                                       "DAYM780201"), nlag = 3, customprops = NULL)
  moreau <- extractMoreauBroto(seq, props = c("CIDH920105", "BHAR880101", "CHAM820101", # Extract Moreau Autocorrelation Features.
                                              "CHAM820102", "CHOC760101", "BIGC670101", "CHAM810101", 
                                              "DAYM780201"), nlag = 3, customprops = NULL)
  out <- c(geary, moran, moreau) # Autocorrelation Descriptor.
  ad <- rbind(ad, out)
  ctdc <- extractCTDC(seq) # Extract Composition Features.
  ctdt <- extractCTDT(seq) # Extract Transition Features.
  ctdd <- extractCTDD(seq) # Extract Distribution Features.
  out <- c(ctdc, ctdt, ctdd) # Composition, Transition, Distribution.
  ctd <- rbind(ctd, out)
  out <- extractCTriad(seq) # Extract Conjoint Triad Features.
  ct <- rbind(ct, out)
  out = PSEAAC(seq, lambda = 11) # Extract Pseudo Amino Acid Composition Features.
  pseaac <- rbind(pseaac, out)
  out <- extractBLOSUM(seq, submat = "AABLOSUM62", k = 5, # Extract BLOSUM62 Features.
                          lag = 3, scale = TRUE)
  blosum <- rbind(blosum, out)
}

