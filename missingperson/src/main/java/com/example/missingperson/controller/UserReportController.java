package com.example.missingperson.controller;

import com.example.missingperson.entity.UserReportEntity;
import com.example.missingperson.service.UserReportService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/reports")
public class UserReportController {

    @Autowired
    private UserReportService userReportService;

    @GetMapping
    public List<UserReportEntity> getAllReports() {
        return userReportService.getAllReports();
    }

    @PostMapping("/add")
    public UserReportEntity addReport(@RequestBody UserReportEntity report) {
        return userReportService.saveReport(report);
    }

    @PutMapping("/{id}")
    public UserReportEntity updateReport(@PathVariable Long id, @RequestBody UserReportEntity report) {
        return userReportService.updateReport(id, report);
    }

    @DeleteMapping("/{id}")
    public void deleteReport(@PathVariable Long id) {
        userReportService.deleteReport(id);
    }
}