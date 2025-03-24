package com.example.missingperson.controller;

import com.example.missingperson.entity.UserEntity;
import com.example.missingperson.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/users")
public class UserController {

    @Autowired
    private UserService userService;

    @GetMapping
    public List<UserEntity> getAllUsers() {
        return userService.getAllUsers();
    }

    @PostMapping("/register")
    public ResponseEntity<?> registerUser(@RequestBody UserEntity user) {
        try {
            System.out.println("Received User: " + user); // Debugging
            UserEntity registeredUser = userService.saveUser(user);
            return ResponseEntity.ok(registeredUser);
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Failed to register user: " + e.getMessage());
        }
    }

    @PostMapping("/login")
    public ResponseEntity<?> loginUser(@RequestBody UserEntity user) {
        String email = user.getEmail();
        String password = user.getPassword();

        UserEntity existingUser = userService.findByEmail(email);
        if (existingUser == null) {
            return ResponseEntity.badRequest().body("Incorrect email");
        }

        if (!existingUser.getPassword().equals(password)) {
            return ResponseEntity.badRequest().body("Incorrect password");
        }

        return ResponseEntity.ok(existingUser);
    }

    @PutMapping("/{id}")
    public UserEntity updateUser(@PathVariable int id, @RequestBody UserEntity user) {
        return userService.updateUser(id, user);
    }

    @DeleteMapping("/{id}")
    public void deleteUser(@PathVariable int id) {
        userService.deleteUser(id);
    }
}