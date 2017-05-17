var express = require("express");
var http = require('http');
var multer  = require('multer');
var bodyParser = require('body-parser');
var app     = express();
var path    = require("path");
var fs = require("fs");
//var sqlite3 = require("sqlite3").verbose();
var exec = require('child_process').exec;
var net = require('net');
var util = require('util');
var upload_file;

app.use(bodyParser());
app.use(express.static(__dirname));

app.post('/upload', function(req, res){    
	console.log("upload");
	res.redirect("/index.html");
});

//app.post('/user_login', function(req, res){    
//    var id = req.query.id;
//    var username_in = req.body.username;
//    var password_in = req.body.password;
//
//});

server = http.createServer(app);
server.listen(8080);
console.log("Running at Port 8080");
