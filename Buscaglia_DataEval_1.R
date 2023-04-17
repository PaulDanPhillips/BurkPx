library(tidyverse)

load('data/Human_BurkPx.rda')
Human_BurkPx %>% pull(SerumID) %>% unique() %>% length()
Human_BurkPx %>% pull(PatientID) %>% unique() %>% length()
Human_BurkPx %>% pull(Status) %>% unique()

# load('data/Human_BurkPx2.rda')
# Human_BurkPx2 %>% pull(SerumID) %>% unique() %>% length()
# Human_BurkPx2 %>% pull(PatientID) %>% unique() %>% length()
# Human_BurkPx2 %>% pull(Status) %>% unique()
#
# load('data/Human_BurkPx_test.rda')
# load('data/Human_BurkPx_test2.rda')
# Human_BurkPx_test2 %>% pull(SerumID) %>% unique() %>% length()
# Human_BurkPx_test2 %>% pull(PatientID) %>% unique() %>% length()
#
# load('data/Human_BurkPx_train.rda')
# load('data/Human_BurkPx_train2.rda')
# Human_BurkPx_train2 %>% pull(SerumID) %>% unique() %>% length()
# Human_BurkPx_train2 %>% pull(PatientID) %>% unique() %>% length()

# The most abundant data set seems to be the Human_BurkPx file, containing 494 unique patient IDs.  This does not
# agree with the written document that there should be a total of n = 500 (400/100).

Burk2 <- Human_BurkPx %>% pivot_wider(names_from = Antigen, values_from = Value)
out <- Burk2 %>% group_by(PatientID, Status) %>% summarize(count = n())

Burk2 %>% pull(TimeGroup) %>% unique()
