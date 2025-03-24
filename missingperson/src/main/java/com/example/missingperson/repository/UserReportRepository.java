package com.example.missingperson.repository;

import com.example.missingperson.entity.UserReportEntity;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserReportRepository extends JpaRepository<UserReportEntity, Long> {
}