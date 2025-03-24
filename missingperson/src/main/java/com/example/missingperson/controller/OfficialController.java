package com.example.missingperson.controller;

import com.example.missingperson.entity.OfficialEntity;
import com.example.missingperson.service.OfficialService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/officials")
public class OfficialController {

    @Autowired
    private OfficialService officialService;

    // Get all officials
    @GetMapping
    public List<OfficialEntity> getAllOfficials() {
        return officialService.getAllOfficials();
    }

    // Register a new official
    @PostMapping("/register")
    public ResponseEntity<?> registerOfficial(@RequestBody OfficialEntity official) {
        try {
            // Check if the email is already registered
            if (officialService.findByEmail(official.getEmail()) != null) {
                return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                        .body(Map.of("success", false, "message", "Email already registered"));
            }

            // Save the official
            OfficialEntity savedOfficial = officialService.saveOfficial(official);
            return ResponseEntity.status(HttpStatus.CREATED)
                    .body(Map.of("success", true, "message", "Official registered successfully", "official", savedOfficial));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("success", false, "message", "Failed to register official: " + e.getMessage()));
        }
    }

    // Login an official
    @PostMapping("/login")
    public ResponseEntity<?> loginOfficial(@RequestBody OfficialEntity official) {
        try {
            OfficialEntity loggedInOfficial = officialService.loginOfficial(official.getEmail(), official.getPassword());
            if (loggedInOfficial != null) {
                return ResponseEntity.ok()
                        .body(Map.of("success", true, "message", "Login successful", "official", loggedInOfficial));
            } else {
                return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                        .body(Map.of("success", false, "message", "Invalid email or password"));
            }
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("success", false, "message", "Failed to log in: " + e.getMessage()));
        }
    }

    // Update an official
    @PutMapping("/{id}")
    public ResponseEntity<?> updateOfficial(@PathVariable int id, @RequestBody OfficialEntity updatedOfficial) {
        try {
            OfficialEntity official = officialService.updateOfficial(id, updatedOfficial);
            return ResponseEntity.ok()
                    .body(Map.of("success", true, "message", "Official updated successfully", "official", official));
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("success", false, "message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("success", false, "message", "Failed to update official: " + e.getMessage()));
        }
    }

    // Delete an official
    @DeleteMapping("/{id}")
    public ResponseEntity<?> deleteOfficial(@PathVariable int id) {
        try {
            officialService.deleteOfficial(id);
            return ResponseEntity.ok()
                    .body(Map.of("success", true, "message", "Official deleted successfully"));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("success", false, "message", "Failed to delete official: " + e.getMessage()));
        }
    }
}