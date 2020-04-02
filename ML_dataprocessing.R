


library(plyr)
library(quantmod)



obs_days = 60 
future_days = 30

df1 = read.csv(file = "./stockproject/data/ML_dataset/america_2020-03-27 (1).csv")
df2 = read.csv(file = "./stockproject/data/ML_dataset/america_2020-03-27 (2).csv")
df3 = read.csv(file = "./stockproject/data/ML_dataset/america_2020-03-27 (3).csv")
df4 = read.csv(file = "./stockproject/data/ML_dataset/america_2020-03-27 (4).csv")
df5 = read.csv(file = "./stockproject/data/ML_dataset/america_2020-03-27 (5).csv")
df6 = read.csv(file = "./stockproject/data/ML_dataset/america_2020-03-27 (6).csv")

df_merge = join(df1, df2, by="Ticker", type="inner")
df_merge = join(df_merge, df3, by="Ticker", type="inner")
df_merge = join(df_merge, df4, by="Ticker", type="inner")
df_merge = join(df_merge, df5, by="Ticker", type="inner")
df_merge = join(df_merge, df6, by="Ticker", type="inner")

subset(df_merge, select=-c(Rating,Ticker))




names = df_merge$Ticker
for (name in names){
  tryCatch({
    df = getSymbols(name, src='yahoo', from = "2019-06-01", to = "2020-01-31", auto.assign = FALSE)  
    df = as.data.frame(df)
    
    if (NROW(df) < obs_days-future_days){
      next
    }
    myFileName = paste("./stockProject/data/ML_dataset/downloadStocks/",name, ".csv", sep = "")
    write.csv(df, file = myFileName, row.names = FALSE)

  }, error=function(e){})
  
}


df_merge[is.na(df_merge)]<-0

df_merge = subset(df_merge, select=-c(Rating, Change.., Change))

write.csv(df_merge, "./stockProject/data/ML_merge.csv", row.names = FALSE)








