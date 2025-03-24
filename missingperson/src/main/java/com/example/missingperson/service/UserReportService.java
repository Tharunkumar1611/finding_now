package com.example.missingperson.service;

import com.example.missingperson.entity.UserReportEntity;
import com.example.missingperson.repository.UserReportRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class UserReportService {

    @Autowired
    private UserReportRepository userReportRepository;

    public List<UserReportEntity> getAllReports() {
        return userReportRepository.findAll();
    }

    public UserReportEntity saveReport(UserReportEntity report) {
        return userReportRepository.save(report);
    }

    public UserReportEntity updateReport(Long id, UserReportEntity report) {
        Optional<UserReportEntity> existingReport = userReportRepository.findById(id);
        if (existingReport.isPresent()) {
            UserReportEntity updatedReport = existingReport.get();
            updatedReport.setDescription(report.getDescription());
            updatedReport.setLocation(report.getLocation());
            updatedReport.setTimestamp(report.getTimestamp());
           
            updatedReport.setComment(report.getComment()); // Update comment
            return userReportRepository.save(updatedReport);
        } else {
            throw new RuntimeException("Report not found with id: " + id);
        }
    }

    public void deleteReport(Long id) {
        userReportRepository.deleteById(id);
    }
}