package com.example.missingperson.service;

import com.example.missingperson.entity.UserEntity;
import com.example.missingperson.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class UserService {

    @Autowired
    private UserRepository userRepository;

    public List<UserEntity> getAllUsers() {
        return userRepository.findAll();
    }

    public UserEntity saveUser(UserEntity user) {
        return userRepository.save(user);
    }

    public UserEntity updateUser(int id, UserEntity updatedUser) {
        Optional<UserEntity> existingUser = userRepository.findById(id);
        if (existingUser.isPresent()) {
            UserEntity user = existingUser.get();
            user.setUsername(updatedUser.getUsername());
            user.setPassword(updatedUser.getPassword());
            user.setEmail(updatedUser.getEmail());
            user.setPhone_number(updatedUser.getPhone_number());
            user.setAddress(updatedUser.getAddress());
            return userRepository.save(user);
        } else {
            throw new RuntimeException("User not found with id: " + id);
        }
    }

    public void deleteUser(int id) {
        userRepository.deleteById(id);
    }

    public UserEntity findByEmail(String email) {
        return userRepository.findByEmail(email);
    }
}