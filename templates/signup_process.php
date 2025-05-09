<?php
// Database connection variables
$servername = "localhost";
$username = "root"; // default user for XAMPP
$password = "";     // default password is blank
$dbname = "agroguard";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Collect form data
$fullname = $_POST['fullname'];
$email = $_POST['email'];
$username = $_POST['username'];
$raw_password = $_POST['password'];
$hashed_password = password_hash($raw_password, PASSWORD_DEFAULT); // encrypt password

// Insert into database
$sql = "INSERT INTO users (fullname, email, username, password)
        VALUES (?, ?, ?, ?)";

$stmt = $conn->prepare($sql);
$stmt->bind_param("ssss", $fullname, $email, $username, $hashed_password);

if ($stmt->execute()) {
    echo "<script>alert('Account created successfully!'); window.location.href='signin.html';</script>";
} else {
    echo "<script>alert('Error: Email or Username already exists.'); window.location.href='signup.html';</script>";
}
if ($conn->query($sql) === TRUE) {
    header("Location: index.html"); // Redirect to homepage
    exit();
} else {
    echo "Error: " . $sql . "<br>" . $conn->error;
}

$conn->close();
?>
$stmt->close();
$conn->close();
?>
