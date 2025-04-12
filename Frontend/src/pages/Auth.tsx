import { useState, ChangeEvent, FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface FormData {
  username: string;
  email: string;
  password: string;
}

export default function Auth() {
  const [isLogin, setIsLogin] = useState<boolean>(true);
  const [formData, setFormData] = useState<FormData>({
    username: "",
    email: "",
    password: "",
  });
  const [message, setMessage] = useState<string>("");
  const [error, setError] = useState<string>("");
  const navigate = useNavigate();

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setMessage("");
    setError("");

    try {
      const endpoint = isLogin ? "/login" : "/signup";
      const payload = isLogin
        ? { username: formData.username, password: formData.password }
        : { username: formData.username, email: formData.email, password: formData.password };

      await axios.post(`http://localhost:8080${endpoint}`, payload, { withCredentials: true });
      if (isLogin) {
        navigate("/");
      } else {
        setMessage("Signup successful! Please log in.");
      }
    } catch (err: any) {
      setError(err.response?.data?.message || "An error occurred");
    }
  };

  const toggleAuthMode = () => {
    setIsLogin(!isLogin);
    setMessage("");
    setError("");
    setFormData({ username: "", email: "", password: "" });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-white dark:bg-black">
      <Card className="w-96 border border-gray-200 dark:border-gray-800 shadow-md">
        <CardHeader>
          <CardTitle className="text-center">{isLogin ? "Login" : "Sign Up"}</CardTitle>
          <CardDescription className="text-center">
            {isLogin ? "Enter your credentials to access your account" : "Create a new account"}
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          {message && (
            <Alert className="mb-4 bg-green-50 text-green-800 dark:bg-green-900/20 dark:text-green-400">
              <AlertDescription>{message}</AlertDescription>
            </Alert>
          )}
          
          {error && (
            <Alert className="mb-4 bg-red-50 text-red-800 dark:bg-red-900/20 dark:text-red-400">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                name="username"
                value={formData.username}
                onChange={handleChange}
                placeholder="Enter your username"
                className="bg-white dark:bg-gray-900"
              />
            </div>

            {!isLogin && (
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  name="email"
                  type="email"
                  value={formData.email}
                  onChange={handleChange}
                  placeholder="your@email.com"
                  className="bg-white dark:bg-gray-900"
                />
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Enter your password"
                className="bg-white dark:bg-gray-900"
              />
            </div>

            <Button type="submit" className="w-full bg-black hover:bg-gray-800 text-white dark:bg-white dark:text-black dark:hover:bg-gray-200">
              {isLogin ? "Login" : "Sign Up"}
            </Button>
          </form>
        </CardContent>
        
        <CardFooter className="flex justify-center">
          <Button 
            variant="link" 
            onClick={toggleAuthMode}
            className="text-black dark:text-white hover:text-gray-600 dark:hover:text-gray-400"
          >
            {isLogin ? "Create an account" : "Already have an account? Login"}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}