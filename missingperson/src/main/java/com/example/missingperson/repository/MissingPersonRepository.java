package com.example.missingperson.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import com.example.missingperson.entity.MissingPersonEntity;

public interface MissingPersonRepository extends JpaRepository<MissingPersonEntity, Long> {
}
