library(ARTool)
library(broom)

df <- read.csv("C:\\Users\\Jake\\Desktop\\python_code\\Lab_Analyses\\Lab_Analyses\\NoteBooks\\test_data.csv")
df$DV_l <- as.factor(df$DV_l)
df$IV3 <- as.factor(df$IV3)

m <- art(DV ~ DV_l + IV3 + DV_l:IV3 + (1 | Group), data = df)
result <- anova(m)
output <- as.data.frame(result, row.names = make.unique(rownames(result)))

posthoc <- art.con(m, "IV3", adjust = "fdr")
post_output <- as.data.frame(posthoc, row.names = make.unique(rownames(posthoc)))
# capture.output(output, file = "C:\\Users\\Jake\\Desktop\\python_code\\Lab_Analyses\\Lab_Analyses\\NoteBooks\\result.csv")
write.csv(output, "C:\\Users\\Jake\\Desktop\\python_code\\Lab_Analyses\\Lab_Analyses\\NoteBooks\\result.csv", row.names = FALSE)
write.csv(post_output, "C:\\Users\\Jake\\Desktop\\python_code\\Lab_Analyses\\Lab_Analyses\\NoteBooks\\posthoc.csv", row.names = FALSE)
