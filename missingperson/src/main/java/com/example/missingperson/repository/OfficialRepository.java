package com.example.missingperson.repository;

import com.example.missingperson.entity.OfficialEntity;
import org.springframework.data.jpa.repository.JpaRepository;

public interface OfficialRepository extends JpaRepository<OfficialEntity, Integer> {
    OfficialEntity findByEmail(String email); // Find an official by email
}