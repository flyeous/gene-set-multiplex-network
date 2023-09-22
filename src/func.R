add.multiplex.coef <- function(res.gsea, GO, 
                               labels = c("P", "C1", "C2", "Multiplex.PageRank",
                                         "PageRank.Jaccard", "gene.set.size")){
    if (!("ID" %in% colnames(GO))){
        print('"ID" is not in the column names of the GO dataframe!')
        return
    }
    GO_ID =  GO$'ID'
    if (!all(labels %in% colnames(GO))){
        print("Not all labels are in the column names of the GO dataframe!")
        return
    }
    
    for (ns in labels){
        item  = list()
        for (index in res.gsea@result$'ID'){
            if (index %in% GO_ID){
                item = append(item, GO[ns][,1][index == GO_ID])
            }else{
                print('Missing value in ' + ns + '!')
                item = append(item, NA)
            }
        }
        res.gsea@result[ns] = unlist(item)
        }
    return(res.gsea)
    }


order.gsea <- function(res.gsea, order = "NES", decreasing = TRUE){
    res.gsea@result = res.gsea@result[order(abs(res.gsea@result[order][,1]), decreasing = decreasing),]
    return(res.gsea)
    }