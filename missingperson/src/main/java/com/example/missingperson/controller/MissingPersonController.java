package com.example.missingperson.controller;

import com.example.missingperson.entity.MissingPersonEntity;
import com.example.missingperson.service.MissingPersonService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/missingpersons")
public class MissingPersonController {

    @Autowired
    private MissingPersonService missingPersonService;

    @GetMapping
    public List<MissingPersonEntity> getAllMissingPersons() {
        return missingPersonService.getAllMissingPersons();
    }

    @PostMapping("/add")
    public MissingPersonEntity addMissingPerson(@RequestBody MissingPersonEntity person) {
        return missingPersonService.saveMissingPerson(person);
    }

    @PutMapping("/{id}")
    public MissingPersonEntity updateMissingPerson(@PathVariable Long id, @RequestBody MissingPersonEntity person) {
        return missingPersonService.updateMissingPerson(id, person);
    }

    @DeleteMapping("/{id}")
    public void deleteMissingPerson(@PathVariable Long id) {
        missingPersonService.deleteMissingPerson(id);
    }
}
