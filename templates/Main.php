<!DOCTYPE html>
<html>
    <title> Intruder Detector V1.0</title>
    <link rel="stylesheet" href="../css/style1.css">
<body>

    <div class="header">
        <img class="float" src="../img/bois.jpg" style="width:200px;height:150px;margin-left:100px;"><center>
        <h1 style="text-shadow:1px 1px 0 #444"> Intruder Detector V1.0 <br></h1></center>
        </div>
        <br>
        <br>
    <center>

        <?php

        require 'config.php';
        
        
     
        $sql = "SELECT id
        FROM authorize
        INTERSECT
        SELECT accs_prsn
        FROM accs_hist ";

        $result = $conn->query($sql);
        
        if ($result->num_rows > 0) {

           echo("true");
           header("location:https://www.youtube.com/watch?v=yAjj7ByyWx0");

        }
         else {
            echo "<script type='text/javascript'>
            alert('Welcome to Intruder Detection V1.0');
            window.location = '../html/home.html';
            </script>";
        }
        
        $conn->close();
        ?>
        
         <a href="phphome.php" class="button update" > Back to ADMIN HOME </a>
    </center>






















</body>
</html>