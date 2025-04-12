const express = require("express");
const app = express();
const mongoose = require("mongoose");
const { User } = require("./model.js");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken"); // Add JWT for token generation

const port = 8080;
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

async function main() {
    await mongoose.connect('mongodb+srv://manik:Tw0yDTt60qwOmGzO@cluster0.px6sa.mongodb.net/databaseName?retryWrites=true&w=majority') // Add database name
        .then(() => {
            console.log("Connected to Database");
        });
}
main();

app.listen(port, () => {
    console.log(`server is listening on ${port}.`);
});

app.post("/login", async (req, res) => {
    const { username, password } = req.body;

    try {
        const user = await User.findOne({ username });
        if (!user) {
            return res.status(404).json({ message: "User not found" });
        }

        const validPassword = await bcrypt.compare(password, user.password);
        if (!validPassword) {
            return res.status(401).json({ message: 'Invalid credentials' });
        }

        // Generate JWT token
        const token = jwt.sign({ userId: user._id }, 'your_secret_key', { expiresIn: '1h' });
        res.json({ message: 'Logged in successfully', token });
    } catch (error) {
        console.error(error);
        res.status(500).send("Internal Server Error");
    }
});

app.post("/signup", async (req, res) => { // Changed endpoint to /signup for clarity
    const { username, email, password } = req.body;

    try {
        const existingUser = await User.findOne({ username });
        if (existingUser) {
            return res.status(400).json({ message: "User already exists" });
        }

        // Hash password
        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(password, salt);

        // Create new user
        const newUser = new User({
            email,
            username,
            password: hashedPassword // Store hashed password
        });

        await newUser.save();
        res.status(201).json({ message: "User created successfully" });
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: "Internal Server Error" });
    }
});