package com.example.missingperson.service;

import com.example.missingperson.entity.OfficialEntity;
import com.example.missingperson.repository.OfficialRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCrypt;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class OfficialService {

    @Autowired
    private OfficialRepository officialRepository;

    // Get all officials
    public List<OfficialEntity> getAllOfficials() {
        return officialRepository.findAll();
    }

    // Find an official by email
    public OfficialEntity findByEmail(String email) {
        return officialRepository.findByEmail(email);
    }

    // Register a new official (with password hashing)
    public OfficialEntity saveOfficial(OfficialEntity official) {
        // Hash the password before saving
        String hashedPassword = BCrypt.hashpw(official.getPassword(), BCrypt.gensalt());
        official.setPassword(hashedPassword);
        return officialRepository.save(official);
    }

    // Update an existing official
    public OfficialEntity updateOfficial(int id, OfficialEntity updatedOfficial) {
        Optional<OfficialEntity> existingOfficial = officialRepository.findById(id);
        if (existingOfficial.isPresent()) {
            OfficialEntity official = existingOfficial.get();
            official.setName(updatedOfficial.getName());

            // Hash the password if it is being updated
            if (updatedOfficial.getPassword() != null && !updatedOfficial.getPassword().isEmpty()) {
                String hashedPassword = BCrypt.hashpw(updatedOfficial.getPassword(), BCrypt.gensalt());
                official.setPassword(hashedPassword);
            }

            official.setEmail(updatedOfficial.getEmail());
            official.setPhoneNumber(updatedOfficial.getPhoneNumber());
            return officialRepository.save(official);
        } else {
            throw new RuntimeException("Official not found with id: " + id);
        }
    }

    // Delete an official by ID
    public void deleteOfficial(int id) {
        officialRepository.deleteById(id);
    }

    // Login an official
    public OfficialEntity loginOfficial(String email, String password) {
        OfficialEntity official = officialRepository.findByEmail(email);
        if (official != null && BCrypt.checkpw(password, official.getPassword())) {
            return official; // Login successful
        }
        return null; // Login failed
    }
}