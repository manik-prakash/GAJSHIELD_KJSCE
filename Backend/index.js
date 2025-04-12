const express = require("express");
const app = express();
const mongoose = require("mongoose");


const port = 8080;
app.use(express.urlencoded({extended : true}));

app.get("/",(req,res)=>{
    res.send("Hello World!");
})

app.listen(port,()=>{
    console.log(`server is listening on ${port}.`);
});